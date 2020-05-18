#include <iostream>
#include <zmq.h>
#include <cassert>
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
    zmq_connect(ampOutIn_socket, privacyAmplificationServer_address);

    std::cout << "Waiting for PrivacyAmplification Server..." << std::endl;
    while (true) {
        zmq_send(ampOutIn_socket, "SYN", 3, 0);
        printf("SYN SENT\n");
        zmq_recv(ampOutIn_socket, ampOutInData, vertical_bytes, 0);
        printf("ACK SENT\n");
        zmq_send(ampOutIn_socket, "ACK", 3, 0);
        
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
