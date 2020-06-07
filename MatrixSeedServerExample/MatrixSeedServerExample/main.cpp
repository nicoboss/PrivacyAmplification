#include <iostream>
#include <zmq.h>
#include <mutex>
#include <atomic>
#include <sstream>

#define TRUE 1
#define FALSE 0
#define TWO_CLIENTS FALSE
#define println(TEXT) printlnStream(std::ostringstream().flush() << TEXT);
const char* address_alice = "tcp://127.0.0.1:45555";
const char* address_bob = "tcp://127.0.0.1:46666";
constexpr int vertical_len = 96;
constexpr int horizontal_len = 160;
constexpr int key_len = 257;
constexpr int vertical_block = vertical_len / 32;
constexpr int horizontal_block = horizontal_len / 32;
constexpr int key_blocks = vertical_block + horizontal_block + 1;
constexpr int desired_block = vertical_block + horizontal_block;
constexpr int desired_len = vertical_len + horizontal_len;
unsigned int* toeplitz_seed = (unsigned int*)malloc(desired_block);
std::atomic<int> aliceReady = 1;
std::atomic<int> bobReady = 1;
std::mutex printlock;

void printlnStream(std::ostream& os) {
    std::ostringstream& ss = dynamic_cast<std::ostringstream&>(os);
    printlock.lock();
    std::cout << ss.str() << std::endl;
    printlock.unlock();
}

void send_alice() {
    void* context_alice = zmq_ctx_new();
    void* MatrixSeedServer_socket_alice = zmq_socket(context_alice, ZMQ_REP);
    while (zmq_bind(MatrixSeedServer_socket_alice, address_alice) != 0) {
        println("Binding to \"" << address_alice << "\" failed! Retrying...");
    }
    char syn[3];
    int32_t rc;

    println("Waiting for Alice...");
    while (true){
        rc = zmq_recv(MatrixSeedServer_socket_alice, syn, 3, 0);
        if (rc != 3 || syn[0] != 'S' || syn[1] != 'Y' || syn[2] != 'N') {
            println("Error receiving SYN! Retrying...");
            continue;
        }
        if (zmq_send(MatrixSeedServer_socket_alice, toeplitz_seed, desired_block * sizeof(unsigned int), 0) != desired_block * sizeof(unsigned int)) {
            println("Error sending data to Alice! Retrying...");
            continue;
        }
        println("Sent seed to Alice");

        aliceReady = 1;
        while (aliceReady != 0) {
            std::this_thread::yield();
        }
    }

    zmq_unbind(MatrixSeedServer_socket_alice, address_alice);
    zmq_close(MatrixSeedServer_socket_alice);
    zmq_ctx_destroy(MatrixSeedServer_socket_alice);
}

void send_bob() {
    void* context_bob = zmq_ctx_new();
    void* MatrixSeedServer_socket_bob = zmq_socket(context_bob, ZMQ_REP);
    while (zmq_bind(MatrixSeedServer_socket_bob, address_bob) != 0) {
        println("Binding to \"" << address_bob << "\" failed! Retrying...");
    }
    char syn[3];
    int32_t rc;

    println("Waiting for Bob...");
    while (true) {
        rc = zmq_recv(MatrixSeedServer_socket_bob, syn, 3, 0);
        if (rc != 3 || syn != "SYN") {
            println("Error receiving SYN! Retrying...");
            continue;
        }
        if (zmq_send(MatrixSeedServer_socket_bob, toeplitz_seed, desired_block * sizeof(unsigned int), 0) != desired_block * sizeof(unsigned int)) {
            println("Error sending data to Bob! Retrying...");
            continue;
        }
        println("Sent seed to Bob");

        bobReady = 1;
        while (bobReady != 0) {
            std::this_thread::yield();
        }
    }

    zmq_unbind(MatrixSeedServer_socket_bob, address_bob);
    zmq_close(MatrixSeedServer_socket_bob);
    zmq_ctx_destroy(MatrixSeedServer_socket_bob);
}


int main(int argc, char* argv[])
{
    std::thread threadReciveObjAlice(send_alice);
    threadReciveObjAlice.detach();
    #if TWO_CLIENTS == TRUE
    std::thread threadReciveObjBob(send_bob);
    threadReciveObjBob.detach();
    #endif

    while (true) {

        #if TWO_CLIENTS == TRUE
        while (aliceReady == 0 || bobReady == 0) {
        #else
        while (aliceReady == 0) {
        #endif
            std::this_thread::yield();
        }

        unsigned int* vertical_data_alice = new unsigned int[vertical_block] { 0b10101000111111011010010101110111, 0b11110011000000000110101010011000, 0b11110110110001111100111001101001 };
        unsigned int* horizontal_data_alice = new unsigned int[horizontal_block] { 0b10010011000111011111001011110011, 0b10111010011101010011011101000100, 0b11111100010001011111010011000100, 0b00110101010111010010000010010111, 0b01001110101110100111110001100101 };
        memcpy(toeplitz_seed, vertical_data_alice, vertical_block * sizeof(unsigned int));
        memcpy(toeplitz_seed + vertical_block, horizontal_data_alice, horizontal_block * sizeof(unsigned int));

        aliceReady = 0;
        bobReady = 0;

    }
    return 0;
}
