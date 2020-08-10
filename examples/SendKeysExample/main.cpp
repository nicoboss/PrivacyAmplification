#define _CRT_SECURE_NO_WARNINGS
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <zmq.h>
using namespace std;

#define fatal(TEXT) fatalPrint(std::ostringstream().flush() << TEXT);
const char* address = "tcp://127.0.0.1:47777";
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
unsigned int* key_data_alice = new unsigned int[key_blocks];


void fatalPrint(ostream& os) {
    cout << dynamic_cast<ostringstream&>(os).str() << endl;
    exit(1);
    abort();
}


size_t getFileSize(ifstream& file) {
    int pos = file.tellg();
    file.seekg(0, ios::end);
    size_t filelength = file.tellg();
    file.seekg(pos, ios::beg);
    return filelength;
}


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
}


int main(int argc, char* argv[])
{
    void* context = zmq_ctx_new();
    void* SendKeys_socket = zmq_socket(context, ZMQ_REP);
    while (zmq_bind(SendKeys_socket, address) != 0) {
        cout << "Binding to \"" << address << "\" failed! Retrying..." << endl;
    }

    //4 time 0x00 bytes at the end for conversion to unsigned int array
    //Key data alice in liddle endians
    fromFile("keyfile.bin", key_data_alice);

    char syn[3];
    int32_t rc;
    cout << "Waiting for clients..." << endl;

    while (true) {
        rc = zmq_recv(SendKeys_socket, syn, 3, 0);
        if (rc != 3 || syn[0] != 'S' || syn[1] != 'Y' || syn[2] != 'N') {
            cout << "Error receiving SYN! Retrying..." << endl;
            continue;
        }
        if (zmq_send(SendKeys_socket, &vertical_block, sizeof(uint32_t), ZMQ_SNDMORE) != sizeof(uint32_t)) {
            cout << "Error sending vertical_blocks! Retrying..." << endl;
            continue;
        }
        if (zmq_send(SendKeys_socket, key_data_alice, key_blocks * sizeof(unsigned int), 0) != key_blocks * sizeof(unsigned int)) {
            cout << "Error sending Key! Retrying..." << endl;
            continue;
        }
        auto currentTime = time(nullptr);
        cout << put_time(localtime(&currentTime), "%F %T") << " Sent Key" << endl;
    }

    zmq_unbind(SendKeys_socket, address);
    zmq_close(SendKeys_socket);
    zmq_ctx_destroy(SendKeys_socket);
    return 0;
}
