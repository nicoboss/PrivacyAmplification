#include <iostream>
#include <zmq.h>
#include <mutex>
#include <bitset>

const char* privacyAmplificationServer_address = "tcp://127.0.0.1:48888";
constexpr int vertical_len = 96;
constexpr int vertical_bytes = vertical_len / 8;
#define min(a,b) (((a) < (b)) ? (a) : (b))
unsigned char* ampOutInData = (unsigned char*)malloc(vertical_bytes);

int main(int argc, char* argv[])
{
    void* context = zmq_ctx_new();
    void* ampOutIn_socket = zmq_socket(context, ZMQ_REQ);
    while (zmq_connect(ampOutIn_socket, privacyAmplificationServer_address) != 0) {
        std::cout << "Connection to \"" << privacyAmplificationServer_address << "\" failed! Retrying..." << std::endl;
    }

    std::cout << "Waiting for PrivacyAmplification Server..." << std::endl;
    while (true) {
        if (zmq_send(ampOutIn_socket, "SYN", 3, 0) != 3) {
            std::cout << "Error sending SYN to PrivacyAmplification Server! Retrying..." << std::endl;
            continue;
        }
        if (zmq_recv(ampOutIn_socket, ampOutInData, vertical_bytes, 0) != vertical_bytes) {
            std::cout << "Error receiving data from PrivacyAmplification Server! Retrying..." << std::endl;
            continue;
        }
        std::cout << "Key Block recived" << std::endl;
        
        for (size_t i = 0; i < min(vertical_bytes, 16); ++i)
        {
            printf("0x%02X: %s\n", ampOutInData[i], std::bitset<8>(ampOutInData[i]).to_string().c_str());
        }
    }

    zmq_unbind(ampOutIn_socket, privacyAmplificationServer_address);
    zmq_close(ampOutIn_socket);
    zmq_ctx_destroy(ampOutIn_socket);
    return 0;
}
