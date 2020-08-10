#define _CRT_SECURE_NO_WARNINGS
#include <iomanip>
#include <iostream>
#include <zmq.h>
#include <mutex>

const char* address = "tcp://127.0.0.1:47777";
constexpr uint32_t vertical_len = 96;
constexpr uint32_t horizontal_len = 160;
constexpr uint32_t key_len = 257;
constexpr uint32_t vertical_block = vertical_len / 32;
constexpr uint32_t horizontal_block = horizontal_len / 32;
constexpr uint32_t key_blocks = vertical_block + horizontal_block + 1;
constexpr uint32_t desired_block = vertical_block + horizontal_block;
constexpr uint32_t desired_len = vertical_len + horizontal_len;


int main(int argc, char* argv[])
{
    void* context = zmq_ctx_new();
    void* SendKeys_socket = zmq_socket(context, ZMQ_REP);
    while (zmq_bind(SendKeys_socket, address) != 0) {
        std::cout << "Binding to \"" << address << "\" failed! Retrying..." << std::endl;
    }

    //4 0x00 bytes at the end for conversion to unsigned int array
    //Key data alice in liddle endians
    //Byte-Array:    0b10010011, 0b01001101, 0b00010101, 0b11110001, 0b01101001, 0b11110000, 0b10111001, 0b00110001, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000
    //Big-Endian:    0b10010011010011010001010111110001, 0b01101001111100001011100100110001, 0b00000000000000000000000000000000
    //Little-Endian: 0b1111000100010101010011011001 0011, 0b00110001101110011111000001101001, 0b00000000000000000000000000000000
    unsigned int* key_data_alice = new unsigned int[key_blocks] { 0b11101100100100001000110111011101, 0b00111111111110111000110111111111, 0b11011100001101001011010100001101, 0b10001010001100000010000010001000, 0b00111100110011010010100100110000, 0b00100111100010001000111110011101, 0b01111110101100011000110010101111, 0b01001110101100010100101100111011, 0b10000000000000000000000000000000 };
    unsigned int* codeword_bin = new unsigned int[key_blocks];

    char syn[3];
    int32_t rc;
    std::cout << "Waiting for clients..." << std::endl;

    while (true) {
        rc = zmq_recv(SendKeys_socket, syn, 3, 0);
        if (rc != 3 || syn[0] != 'S' || syn[1] != 'Y' || syn[2] != 'N') {
            std::cout << "Error receiving SYN! Retrying..." << std::endl;
            continue;
        }
        if (zmq_send(SendKeys_socket, reinterpret_cast<const char*>(&vertical_block), sizeof(uint32_t), ZMQ_SNDMORE) != sizeof(uint32_t)) {
            std::cout << "Error sending vertical_blocks! Retrying..." << std::endl;
            continue;
        }
        if (zmq_send(SendKeys_socket, key_data_alice, key_blocks * sizeof(unsigned int), 0) != key_blocks * sizeof(unsigned int)) {
            std::cout << "Error sending Key! Retrying..." << std::endl;
            continue;
        }
        auto time = std::time(nullptr);
        std::cout << std::put_time(std::localtime(&time), "%F %T") << " Sent Key" << std::endl;
    }

    zmq_unbind(SendKeys_socket, address);
    zmq_close(SendKeys_socket);
    zmq_ctx_destroy(SendKeys_socket);
    return 0;
}
