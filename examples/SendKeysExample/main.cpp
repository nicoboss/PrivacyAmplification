#define _CRT_SECURE_NO_WARNINGS
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <mutex>
#include <zmq.h>
using namespace std;

/*Prints an error then terminates with error code 01.
  fatal("Can't open file \"" << filepath << "\" => terminating!");*/
#define fatal(TEXT) fatalPrint(std::ostringstream().flush() << TEXT);

/*Address where the key server should bind to.*/
const char* address = "tcp://127.0.0.1:47777";

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
unsigned int* key_data = new unsigned int[key_blocks];
constexpr bool pa_do_xor_key_rest = true;
constexpr bool pa_do_compress = true;


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


/// @brief Gets the raw corrected key from a file
/// @param[in] Input file path
/// @param[out] Buffer where to store the raw corrected key
/// Note: In a real environment this should be connected with
/// the FPGA board receiving and correcting the raw keys.
void fromFile(const char* filepath, unsigned int* recv_key) {
	ifstream keyfile(filepath, ios::binary);

	if (keyfile.fail())
	{
		fatal("Can't open file \"" << filepath << "\" => terminating!");
	}

	size_t keyfile_length = getFileSize(keyfile);
	if (keyfile_length < key_blocks * sizeof(uint32_t))
	{
		fatal("File \"" << filepath << "\" is with " << keyfile_length << " bytes too short!" << endl <<
			"it is required to be at least " << key_blocks * sizeof(uint32_t) << " bytes => terminating!");
	}

	char* recv_key_char = reinterpret_cast<char*>(recv_key);
	keyfile.read(recv_key_char, key_blocks * sizeof(uint32_t));
	keyfile.close();
}


/// @brief Sends the raw corrected keydata to its client
int main(int argc, char* argv[])
{
	void* context = zmq_ctx_new();
	void* SendKeys_socket = zmq_socket(context, ZMQ_PUSH);
	int hwm = 1;
	zmq_setsockopt(SendKeys_socket, ZMQ_SNDHWM, &hwm, sizeof(int));
	while (zmq_bind(SendKeys_socket, address) != 0) {
		cout << "Binding to \"" << address << "\" failed! Retrying..." << endl;
	}
	time_t currentTime;

	cout << "Waiting for clients..." << endl;

	//4 time 0x00 bytes at the end for conversion to unsigned int array
	//Key data alice in little endians
	fromFile("keyfile.bin", key_data);

	while (true) {
		if (zmq_send(SendKeys_socket, &pa_do_xor_key_rest, sizeof(bool), ZMQ_SNDMORE) != sizeof(bool)) {
			cout << "[Key ] Error sending do_xor_key_rest! Retrying..." << endl;
			continue;	
		}
		if (zmq_send(SendKeys_socket, &pa_do_compress, sizeof(bool), ZMQ_SNDMORE) != sizeof(bool)) {
			cout << "[Key ] Error sending do_compress! Retrying..." << endl;
			continue;
		}
		if (zmq_send(SendKeys_socket, &vertical_block, sizeof(uint32_t), ZMQ_SNDMORE) != sizeof(uint32_t)) {
			cout << "Error sending vertical_blocks! Retrying..." << endl;
			continue;
		}
		if (zmq_send(SendKeys_socket, key_data, key_blocks * sizeof(unsigned int), 0) != key_blocks * sizeof(unsigned int)) {
			cout << "Error sending Key! Retrying..." << endl;
			continue;
		}
		time(&currentTime);
		cout << put_time(localtime(&currentTime), "%F %T") << " Sent Key" << endl;
	}

	zmq_unbind(SendKeys_socket, address);
	zmq_close(SendKeys_socket);
	zmq_ctx_destroy(SendKeys_socket);
	return 0;
}
