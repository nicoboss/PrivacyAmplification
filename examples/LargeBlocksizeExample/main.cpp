#define _CRT_SECURE_NO_WARNINGS
#include <iomanip>
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <time.h>
#include <thread>
#include <mutex>
#include <bitset>
#include <atomic>
#include <zmq.h>
#include "Matrix.h"
#include "sha3/sha3.h"
using namespace std;

#define TRUE 1
#define FALSE 0

/*Prints a stream in a thread safe way and adds a newline at the end.
  Example: println("Hallo " << name)*/
#define println(TEXT) printlnStream(ostringstream().flush() << TEXT);

/*Prints an error then terminates with error code 01.
  fatal("Can't open file \"" << filepath << "\" => terminating!");*/
#define fatal(TEXT) fatalPrint(ostringstream().flush() << TEXT);

/*Address where the Server for the first client should bind to.*/
const char* address_seed_server = "tcp://127.0.0.1:45555";

/*Address where the key server should bind to.*/
const char* address_key_server = "tcp://127.0.0.1:47777";

/*Address of the Privacy Amplification Server*/
const char* privacyAmplificationServer_address = "tcp://127.0.0.1:48888";

/*Privacy Amplification input size in bits
  Has to be 2^x and 2^27 is the maximum
  Needs to match with the one specified in other components*/
#define factor 14
#define factor_chunk 11
#define pwrtwo(x) (1 << (x))
#define sample_size pwrtwo(factor)
constexpr uint32_t chunk_size = pwrtwo(factor_chunk);
constexpr uint32_t chunk_side = chunk_size / 2;
#define min(a,b) (((a) < (b)) ? (a) : (b))

constexpr uint32_t vertical_len = sample_size / 4 + sample_size / 8;
constexpr uint32_t horizontal_len = sample_size / 2 + sample_size / 8;
constexpr uint32_t key_len = sample_size + 1;
constexpr uint32_t vertical_block = vertical_len / 32;
constexpr uint32_t horizontal_block = horizontal_len / 32;
constexpr uint32_t key_blocks = vertical_block + horizontal_block + 1;
constexpr uint32_t desired_block = vertical_block + horizontal_block;
constexpr uint32_t desired_len = vertical_len + horizontal_len;
constexpr bool pa_do_xor_key_rest = false;
constexpr bool pa_do_compress = false;

uint32_t* toeplitz_seed = reinterpret_cast<uint32_t*>(malloc(desired_block * sizeof(uint32_t)));
int32_t reuseSeedAmount = 0;
uint32_t* key_data = new uint32_t[key_blocks];
constexpr uint32_t vertical_bytes = vertical_len / 8;
uint8_t* ampOutInData = reinterpret_cast<uint8_t*>(malloc(chunk_side / 8));
uint32_t* ampOutInData_U32 = reinterpret_cast<uint32_t*>(ampOutInData);

constexpr uint32_t chunk_size_blocks = chunk_size / 32;
constexpr uint32_t chunk_side_blocks = chunk_side / 32;
constexpr uint32_t chunk_side_blocks_two = chunk_side_blocks * 2;
constexpr uint32_t chunk_vertical_len = chunk_side / 2;
constexpr uint32_t vertical_chunks = vertical_len / chunk_side;
constexpr uint32_t horizontal_chunks = horizontal_len / chunk_side;

uint32_t* local_seed = reinterpret_cast<uint32_t*>(calloc(2*chunk_side_blocks, sizeof(uint32_t)));

//local_key_padded must use calloc!
//4 time 0x00 bytes at the end for conversion to unsigned int array
uint32_t* local_key_padded = reinterpret_cast<uint32_t*>(calloc(2*chunk_side_blocks+1, sizeof(uint32_t)));

uint8_t* amp_out_arr = reinterpret_cast<uint8_t*>(calloc(vertical_chunks*(chunk_side / 8), sizeof(uint8_t)));

atomic<int> seedServerReady = 1;
atomic<int> keyServerReady = 1;
atomic<int> nextSeed = 1;
atomic<int> nextKey = 1;
mutex printlock;

#ifdef DEBUG
#ifdef _WIN32
#define BREAK __debugbreak();
#else
#define BREAK __builtin_trap();
#endif
#else
#define BREAK
#endif

#define assertTrue(actual) \
if (!(actual)) { \
	std::cerr << "AssertTrueError in function " << __func__ << " in " << __FILE__ << ":" << __LINE__ << endl; \
	BREAK \
	exit(101); \
}

#define GetLocalSeed \
memcpy(local_seed, toeplitz_seed + r + chunk_side_blocks, chunk_side / 8); \
memcpy(local_seed + chunk_side_blocks, toeplitz_seed + r, chunk_side / 8); \
//printBin(local_seed, local_seed+2*chunk_side_blocks);

//local_key_padded must use calloc!
//ToDo: Komentar!
#define GetLocalKey \
uint32_t* key_data_offset = key_data+keyNr*(chunk_side / 32); \
local_key_padded[0] = key_data_offset[0] >> 1; \
for (uint32_t i = 1; i < chunk_side / 32; ++i) { \
	local_key_padded[i] = (((key_data_offset[i - 1] & 1) << 31) | (key_data_offset[i] >> 1)); \
} \
local_key_padded[(chunk_side / 32)] = ((key_data_offset[(chunk_side / 32) - 1] & 1) << 31); \
//printBin(local_key_padded, local_key_padded+2*chunk_side_blocks);

#define XorWithRow \
for (uint8_t i = 0; i < chunk_side / 8; ++i) \
{ \
    amp_out_arr[currentRowNr*(chunk_side / 8)+i] ^= ampOutInData[i]; \
}


#define RecieveAmpOut \
rc = zmq_recv(ampOutIn_socket, ampOutInData, chunk_side / 8, 0); \
if (rc != chunk_side / 8) { \
	cout << "Error receiving data from PrivacyAmplification Server!" << endl; \
	cout << "Expected " << chunk_side / 8 << " bytes but received " << rc << " bytes! Retrying..." << endl; \
	goto reconnect; \
} \
 \
time(&currentTime); \
cout << put_time(localtime(&currentTime), "%F %T") << " Key Block recived" << endl; \



/// @brief Prints a stream in a thread safe way and adds a newline at the end.
/// @param[in] std::ostringstream().flush() << TEXT but used like println(TEXT) with macros
/// Aquires the printlock, prints the provided key, adds a std::endl and
/// releases the printlock. It flushes with the std::endl.
/// Note: This function is supposed to always be called using the println macro.
/// Example: println("Hallo " << name);
void printlnStream(ostream& os) {
	ostringstream& ss = dynamic_cast<ostringstream&>(os);
	printlock.lock();
	cout << ss.str() << endl;
	printlock.unlock();
}


/// @brief Prints an error then terminates with error code 01.
/// @param[in] std::ostringstream().flush() << TEXT but used like fatal(TEXT) with macros
/// Note: This function is supossed to always be called using the fatal macro.
/// Example: fatal("Can't open file \"" << filepath << "\" => terminating!");*/
void fatalPrint(ostream& os) {
	cout << dynamic_cast<ostringstream&>(os).str() << endl;
	exit(1);
	abort();
}


void printBin(const uint8_t* position, const uint8_t* end) {
	while (position < end) {
		printf("%s", bitset<8>(*position).to_string().c_str());
		++position;
	}
	cout << endl;
}


void printBin(const uint32_t* position, const uint32_t* end) {
	while (position < end) {
		printf("%s", bitset<32>(*position).to_string().c_str());
		++position;
	}
	cout << endl;
}

string toHexString(const uint8_t* data, uint32_t data_length) {
	std::stringstream ss;
	ss << "{ ";
	for (int i = 0; i < data_length; ++i) {
		ss << std::uppercase << std::hex << "0x" << std::setw(2) << std::setfill('0') << (int)data[i];
		(i % 8 == 7 && i + 1 < data_length)
			? ss << "," << std::endl << "  "
			: ss << ", ";
	}
	ss.seekp(-2, std::ios_base::end);
	ss << " };";
	return ss.str();
}

bool isSha3(const uint8_t* dataToVerify, uint32_t dataToVerify_length, const uint8_t expectedHash[]) {
	sha3_ctx sha3;
	rhash_sha3_256_init(&sha3);
	rhash_sha3_update(&sha3, dataToVerify, dataToVerify_length);
	uint8_t* calculatedHash = (uint8_t*)malloc(32);
	rhash_sha3_final(&sha3, calculatedHash);
	println(toHexString(calculatedHash, 32));
	return memcmp(calculatedHash, expectedHash, 32) == 0;
}


/// @brief Gets the file size from a input file stream
/// @param[in] Input file stream from which the size should get determinated
/// @return size of provided input file stream
size_t getFileSize(ifstream& file) {
	int pos = file.tellg();
	file.seekg(0, ios::end);
	size_t filelength = file.tellg();
	file.seekg(pos, ios::beg);
	return filelength;
}


void binTo4byteLittleEndian(uint8_t* in_arg, uint32_t* out, uint32_t inSize) {
	uint32_t outPos = 0;
	for (uint32_t inPos = 0; inPos < inSize; inPos += 32) {
		uint8_t* in = in_arg + inPos;
		out[outPos] = (((in[0x00] & 1) << 31) | ((in[0x01] & 1) << 30) | ((in[0x02] & 1) << 29) | ((in[0x03] & 1) << 28) | ((in[0x04] & 1) << 27) | ((in[0x05] & 1) << 26) | ((in[0x06] & 1) << 25) | ((in[0x07] & 1) << 24)) |
					  (((in[0x08] & 1) << 23) | ((in[0x09] & 1) << 22) | ((in[0x0A] & 1) << 21) | ((in[0x0B] & 1) << 20) | ((in[0x0C] & 1) << 19) | ((in[0x0D] & 1) << 18) | ((in[0x0E] & 1) << 17) | ((in[0x0F] & 1) << 16)) |
					  (((in[0x10] & 1) << 15) | ((in[0x11] & 1) << 14) | ((in[0x12] & 1) << 13) | ((in[0x13] & 1) << 12) | ((in[0x14] & 1) << 11) | ((in[0x15] & 1) << 10) | ((in[0x16] & 1) << 9) | ((in[0x17] & 1) << 8)) |
					  (((in[0x18] & 1) << 7) | ((in[0x19] & 1) << 6) | ((in[0x1A] & 1) << 5) | ((in[0x1B] & 1) << 4) | ((in[0x1C] & 1) << 3) | ((in[0x1D] & 1) << 2) | ((in[0x1E] & 1) << 1) | ((in[0x1F] & 1) << 0));
		++outPos;
	}
}


/// @brief Send data to the Clients connected to the matrix seed server.
void sendSeed() {
	void* context = zmq_ctx_new();
	void* MatrixSeedServer_socket = zmq_socket(context, ZMQ_PUSH);
	int hwm = 1;
	zmq_setsockopt(MatrixSeedServer_socket, ZMQ_SNDHWM, &hwm, sizeof(int));
	while (zmq_bind(MatrixSeedServer_socket, address_seed_server) != 0) {
		println("Binding to \"" << address_seed_server << "\" failed! Retrying...");
	}
	int32_t rc;
	time_t currentTime;

	println("[Seed] Waiting for Client...");
	while (true) {
		if (zmq_send(MatrixSeedServer_socket, &reuseSeedAmount, sizeof(int32_t), ZMQ_SNDMORE) != sizeof(int32_t)) {
			println("[Seed] Error sending reuseSeedAmount! Retrying...");
			continue;
		}
		if (zmq_send(MatrixSeedServer_socket, local_seed, chunk_size_blocks * sizeof(uint32_t), 0) != chunk_size_blocks * sizeof(uint32_t)) {
			println("[Seed] Error sending seed! Retrying...");
			continue;
		}
		time(&currentTime);
		println("[Seed] " << std::put_time(localtime(&currentTime), "%F %T") << " Sent Seed");

		seedServerReady = 1;
		while (seedServerReady != 0) {
			this_thread::yield();
		}
	}

	zmq_unbind(MatrixSeedServer_socket, address_seed_server);
	zmq_close(MatrixSeedServer_socket);
	zmq_ctx_destroy(MatrixSeedServer_socket);
}


void sendKey() {
	void* context = zmq_ctx_new();
	void* SendKeys_socket = zmq_socket(context, ZMQ_PUSH);
	int hwm = 1;
	zmq_setsockopt(SendKeys_socket, ZMQ_SNDHWM, &hwm, sizeof(int));
	while (zmq_bind(SendKeys_socket, address_key_server) != 0) {
		println("[Key ] Binding to \"" << address_key_server << "\" failed! Retrying...");
	}

	int32_t rc;
	time_t currentTime;
	println("[Key ] Waiting for clients...");

	while (true) {
		if (zmq_send(SendKeys_socket, &pa_do_xor_key_rest, sizeof(bool), ZMQ_SNDMORE) != sizeof(bool)) {
			println("[Key ] Error sending do_xor_key_rest! Retrying...");
			continue;
		}
		if (zmq_send(SendKeys_socket, &pa_do_compress, sizeof(bool), ZMQ_SNDMORE) != sizeof(bool)) {
			cout << "[Key ] Error sending do_compress! Retrying..." << endl;
			continue;
		}
		if (zmq_send(SendKeys_socket, &chunk_side_blocks, sizeof(uint32_t), ZMQ_SNDMORE) != sizeof(uint32_t)) {
			println("[Key ] Error sending vertical_blocks! Retrying...");
			continue;
		}
		if (zmq_send(SendKeys_socket, local_key_padded, (chunk_side_blocks + 1) * sizeof(uint32_t), 0) != (chunk_side_blocks + 1) * sizeof(uint32_t)) {
			println("[Key ] Error sending Key! Retrying...");
			continue;
		}
		time(&currentTime);
		println("[Key ] " << put_time(localtime(&currentTime), "%F %T") << " Sent Key");

		keyServerReady = 1;
		while (keyServerReady != 0) {
			this_thread::yield();
		}
	}

	zmq_unbind(SendKeys_socket, address_key_server);
	zmq_close(SendKeys_socket);
	zmq_ctx_destroy(SendKeys_socket);
}


void seedProvider()
{
	binTo4byteLittleEndian(toeplitz_seed_input, toeplitz_seed, desired_len);

	while (true)
	{
		uint32_t currentRowNr = 0;
		uint32_t rNr = 0;
		uint32_t r = 0;

		for (int32_t columnNr = horizontal_chunks - 1; columnNr > -1; --columnNr)
		{
			currentRowNr = 0;
			for (int32_t keyNr = columnNr; keyNr < columnNr + min((horizontal_chunks - 1) - columnNr + 1, vertical_chunks); ++keyNr)
			{
				while (nextSeed == 0) {
					this_thread::yield();
				}
				nextSeed = 0;
				if (keyNr == columnNr) {
					while (seedServerReady == 0) {
						this_thread::yield();
					}
					GetLocalSeed;
					reuseSeedAmount = min((horizontal_chunks - 1) - columnNr + 1, vertical_chunks);
					seedServerReady = 0;
				}
				++currentRowNr;
			}
			r += chunk_side_blocks;
			++rNr;
		}

		for (int32_t rowNr = 1; rowNr < vertical_chunks; ++rowNr)
		{
			currentRowNr = rowNr;
			for (int32_t keyNr = 0; keyNr < min(horizontal_len / chunk_side_blocks, (vertical_chunks - rowNr)); ++keyNr)
			{
				while (nextSeed == 0) {
					this_thread::yield();
				}
				nextSeed = 0;
				if (keyNr == 0) {
					while (seedServerReady == 0) {
						this_thread::yield();
					}
					GetLocalSeed;
					reuseSeedAmount = min(horizontal_len / chunk_side_blocks, (vertical_chunks - rowNr));
					seedServerReady = 0;
				}
				++currentRowNr;
			}
			r += chunk_side_blocks;
			++rNr;
		}
	}
}

void keyProvider()
{
	//4 time 0x00 bytes at the end for conversion to unsigned int array
	//Key data alice in little endians
	binTo4byteLittleEndian(key_input, key_data, desired_len);

	while (true)
	{
		uint32_t currentRowNr = 0;
		uint32_t rNr = 0;
		uint32_t r = 0;

		if (horizontal_chunks <= 1) {
			println("Fatal error: horizontal_chunks <= 1");
			exit(418); //I'm a teapot
		}
		for (int32_t columnNr = horizontal_chunks - 1; columnNr > -1; --columnNr)
		{
			currentRowNr = 0;
			for (int32_t keyNr = columnNr; keyNr < columnNr + min((horizontal_chunks - 1) - columnNr + 1, vertical_chunks); ++keyNr)
			{
				while (keyServerReady == 0) {
					this_thread::yield();
				}
				while (nextKey == 0) {
					this_thread::yield();
				}
				nextKey = 0;
				//println("Key to generate: " << currentRowNr << ", " << keyNr);
				GetLocalKey;
				keyServerReady = 0;
				++currentRowNr;
			}
			r += chunk_side_blocks;
			++rNr;
		}

		for (int32_t rowNr = 1; rowNr < vertical_chunks; ++rowNr)
		{
			currentRowNr = rowNr;
			for (int32_t keyNr = 0; keyNr < min(horizontal_len / chunk_side_blocks, (vertical_chunks - rowNr)); ++keyNr)
			{
				while (keyServerReady == 0) {
					this_thread::yield();
				}
				while (nextKey == 0) {
					this_thread::yield();
				}
				nextKey = 0;
				//println("Key to generate: " << currentRowNr << ", " << keyNr);
				GetLocalKey;
				keyServerReady = 0;
				++currentRowNr;
			}
			r += chunk_side_blocks;
			++rNr;
		}
	}
}

/// @brief Receives data from the Privacy Amplification Sevrer
/// Contains all the comunication and error handling required
/// to receive the privacy amplification results from the Privacy Amplification Server.
/// Currently it prints the first 4 bytes of the result on the screen.
/// Note: In a real environment this code should be integrated in whatever
/// tool makes use of the Privacy Amplification result.
void receiveAmpOut()
{
	int32_t rc;
	time_t currentTime;
	void* context = zmq_ctx_new();

	reconnect:;
	void* ampOutIn_socket = zmq_socket(context, ZMQ_PULL);
	int hwm = 1;
	zmq_setsockopt(ampOutIn_socket, ZMQ_RCVHWM, &hwm, sizeof(int));

	println("Waiting for PrivacyAmplification Server...");
	zmq_connect(ampOutIn_socket, privacyAmplificationServer_address);

	uint32_t ampOutInData_hash_index = 0;
	uint32_t amp_out_intermediate_hash_index = 0;

	while (true)
	{
		uint32_t currentRowNr = 0;
		uint32_t rNr = 0;
		uint32_t r = 0;

		for (int32_t columnNr = horizontal_chunks - 1; columnNr > -1; --columnNr)
		{
			currentRowNr = 0;
			assertTrue(isSha3(reinterpret_cast<uint8_t*>(amp_out_arr), vertical_bytes, amp_out_intermediate_hash[amp_out_intermediate_hash_index++]));
			for (int32_t keyNr = columnNr; keyNr < columnNr + min((horizontal_chunks - 1) - columnNr + 1, vertical_chunks); ++keyNr)
			{
				nextSeed = 1;
				nextKey = 1;
				RecieveAmpOut;
				//printBin(ampOutInData, ampOutInData + chunk_side / 8);
				assertTrue(isSha3(reinterpret_cast<uint8_t*>(ampOutInData), chunk_side / 8, ampOutInData_hash[ampOutInData_hash_index++]));
				XorWithRow;
				//printBin(amp_out_arr, amp_out_arr + vertical_bytes);
				assertTrue(isSha3(reinterpret_cast<uint8_t*>(amp_out_arr), vertical_bytes, amp_out_intermediate_hash[amp_out_intermediate_hash_index++]));
				//system("pause");
				++currentRowNr;
			}
			r += chunk_side_blocks;
			++rNr;
		}

		for (int32_t rowNr = 1; rowNr < vertical_chunks; ++rowNr)
		{
			currentRowNr = rowNr;
			for (int32_t keyNr = 0; keyNr < min(horizontal_len / chunk_side_blocks, (vertical_chunks - rowNr)); ++keyNr)
			{
				nextSeed = 1;
				nextKey = 1;
				RecieveAmpOut;
				//printBin(ampOutInData, ampOutInData + chunk_side / 8);
				assertTrue(isSha3(reinterpret_cast<uint8_t*>(ampOutInData), chunk_side / 8, ampOutInData_hash[ampOutInData_hash_index++]));
				XorWithRow;
				assertTrue(isSha3(reinterpret_cast<uint8_t*>(amp_out_arr), vertical_bytes, amp_out_intermediate_hash[amp_out_intermediate_hash_index++]));
				//printBin(amp_out_arr, amp_out_arr + vertical_bytes);
				++currentRowNr;
			}
			r += chunk_side_blocks;
			++rNr;
		}
		println("Pre XOR:");
		printBin(amp_out_arr, amp_out_arr + vertical_bytes);
		assertTrue(isSha3(amp_out_arr, vertical_bytes, amp_out_pre_hash));
		uint8_t* key_rest = reinterpret_cast<uint8_t*>(key_data) + horizontal_len / 8;
		println("Key Rest:");
		printBin(key_rest, key_rest + vertical_bytes);
		assertTrue(isSha3(key_rest, vertical_bytes, key_rest_hash));
		for (int32_t i = 0; i < vertical_bytes; i+=4)
		{
			amp_out_arr[i] ^= key_rest[i+3];
			amp_out_arr[i+1] ^= key_rest[i+2];
			amp_out_arr[i+2] ^= key_rest[i+1];
			amp_out_arr[i+3] ^= key_rest[i];
		}
		println("Result:");
		printBin(amp_out_arr, amp_out_arr + vertical_bytes);
		assertTrue(isSha3(amp_out_arr, vertical_bytes, amp_out_reult));
		memset(amp_out_arr, 0x00, vertical_chunks * (chunk_side_blocks) * sizeof(uint32_t));
		break;
	}
	zmq_close(ampOutIn_socket);
	zmq_ctx_destroy(ampOutIn_socket);
	exit(0);
	abort();
}


/// @brief The main function cordinates all servers
/// Ensures that all servers have serverd thair data to its clients 
/// before switching to the next toeplitz matrix seed
int main(int argc, char* argv[])
{
	if (sample_size / 8 < chunk_side) {
		println("Fatal error: sample_size/8 < chunk_side");
		exit(418); //I'm a teapot
	}
	thread threadSeedProvider(seedProvider);
	threadSeedProvider.detach();
	thread threadKeyProvider(keyProvider);
	threadKeyProvider.detach();
	thread threadSendSeed(sendSeed);
	threadSendSeed.detach();
	thread threadSendKey(sendKey);
	threadSendKey.detach();
	thread threadReceiveAmpOut(receiveAmpOut);
	threadReceiveAmpOut.detach();
	while(true) {
		std::this_thread::sleep_for(10s);
		//println("[Stat] Still alive");
	}
	return 0;
}
