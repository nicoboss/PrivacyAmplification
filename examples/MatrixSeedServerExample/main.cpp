#define _CRT_SECURE_NO_WARNINGS
#include <iomanip>
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <time.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <zmq.h>
using namespace std;

#define TRUE 1
#define FALSE 0

/*Specifies if one or two clients should be served*/
#define TWO_CLIENTS FALSE

/*Prints a stream in a thread safe way and adds a newline at the end.
  Example: println("Hallo " << name)*/
#define println(TEXT) printlnStream(ostringstream().flush() << TEXT);

/*Prints an error then terminates with error code 01.
  fatal("Can't open file \"" << filepath << "\" => terminating!");*/
#define fatal(TEXT) fatalPrint(ostringstream().flush() << TEXT);

/*Address where the Server for the first client should bind to.*/
const char* address_alice = "tcp://127.0.0.1:45555";

/*Address where the Server for the second client should bind to.*/
const char* address_bob = "tcp://127.0.0.1:46666";

/*Privacy Amplification input size in bits
  Has to be 2^x and 2^27 is the maximum
  Needs to match with the one specified in other components*/
#define factor 27
#define pwrtwo(x) (1 << (x))
#define sample_size pwrtwo(factor)

constexpr uint32_t vertical_len = sample_size / 4 + sample_size / 8;
constexpr uint32_t horizontal_len = sample_size / 2 + sample_size / 8;
constexpr uint32_t key_len = sample_size + 1;
constexpr uint32_t vertical_block = vertical_len / 32;
constexpr uint32_t horizontal_block = horizontal_len / 32;
constexpr uint32_t key_blocks = vertical_block + horizontal_block + 1;
constexpr uint32_t desired_block = vertical_block + horizontal_block;
constexpr uint32_t desired_len = vertical_len + horizontal_len;
unsigned int* toeplitz_seed = (unsigned int*)malloc(desired_block * sizeof(uint32_t));

atomic<int> aliceReady = 1;
atomic<int> bobReady = 1;
mutex printlock;


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


/// @brief Gets one toeplitz matrix seed from a file
/// @param[in] Input file path
/// @param[out] Buffer where to store the toeplitz matrix seed
/// Note: In a real environment this should be connected with
/// a Cryptograpicaly secure random number generator.
void fromFile(const char* filepath, unsigned int* recv_seed) {
	ifstream seedfile(filepath, ios::binary);

	if (seedfile.fail())
	{
		fatal("Can't open file \"" << filepath << "\" => terminating!");
	}

	size_t seedfile_length = getFileSize(seedfile);
	if (seedfile_length < desired_block * sizeof(uint32_t))
	{
		fatal("File \"" << filepath << "\" is with " << seedfile_length << " bytes too short!" << endl <<
			"it is required to be at least " << desired_block * sizeof(uint32_t) << " bytes => terminating!");
	}

	char* recv_seed_char = reinterpret_cast<char*>(recv_seed);
	seedfile.read(recv_seed_char, desired_block * sizeof(uint32_t));
	seedfile.close();
}


/// @brief Send data to the Clients connected to the matrix seed server.
void sendData(const char* address, atomic<int>* ready, const char* client_name) {
	void* context = zmq_ctx_new();
	void* MatrixSeedServer_socket = zmq_socket(context, ZMQ_REP);
	while (zmq_bind(MatrixSeedServer_socket, address) != 0) {
		println("Binding to \"" << address << "\" failed! Retrying...");
	}
	char syn[3];
	int32_t rc;
	time_t currentTime;

	println("Waiting for Client " << client_name << " ...");
	while (true) {
		rc = zmq_recv(MatrixSeedServer_socket, syn, 3, 0);
		if (rc != 3 || syn[0] != 'S' || syn[1] != 'Y' || syn[2] != 'N') {
			println("Error receiving SYN! Retrying...");
			continue;
		}
		if (zmq_send(MatrixSeedServer_socket, toeplitz_seed, desired_block * sizeof(unsigned int), 0) != desired_block * sizeof(unsigned int)) {
			println("Error sending data to " << client_name << "! Retrying...");
			continue;
		}
		time(&currentTime);
		println(std::put_time(localtime(&currentTime), "%F %T") << " Sent seed to Client " << client_name);

		*ready = 1;
		while (*ready != 0) {
			this_thread::yield();
		}
	}

	zmq_unbind(MatrixSeedServer_socket, address);
	zmq_close(MatrixSeedServer_socket);
	zmq_ctx_destroy(MatrixSeedServer_socket);
}


/// @brief The main function cordinates all servers
/// Ensures that all servers have serverd thair data to its clients 
/// before switching to the next toeplitz matrix seed
int main(int argc, char* argv[])
{
	thread threadSendDataAlice(sendData, address_alice, &aliceReady, "Alice");
	threadSendDataAlice.detach();
	#if TWO_CLIENTS == TRUE
	thread threadSendDataBob(sendData, address_bob, &bobReady, "Bob");
	threadSendDataBob.detach();
	#endif

	while (true) {

		#if TWO_CLIENTS == TRUE
		while (aliceReady == 0 || bobReady == 0) {
		#else
		while (aliceReady == 0) {
		#endif
			this_thread::yield();
		}

		fromFile("toeplitz_seed.bin", toeplitz_seed);

		aliceReady = 0;
		bobReady = 0;

	}
	return 0;
}
