#include <iostream>
#include <zmq.h>
#include <cassert>
#include <mutex>
#include <bitset>

const char* privacyAmplificationServer_address = "tcp://127.0.0.1:48888";
#define sample_size 256
#define min(a,b) (((a) < (b)) ? (a) : (b))
unsigned int* ampOutInData = (unsigned int*)malloc(sample_size / 8);

int main(int argc, char* argv[])
{
    void* context = zmq_ctx_new();
    void* ampOutIn_socket = zmq_socket(context, ZMQ_REQ);
    zmq_connect(ampOutIn_socket, privacyAmplificationServer_address);

    std::cout << "Waiting for PrivacyAmplification Server..." << std::endl;
    while (true) {
        zmq_send(ampOutIn_socket, "SYN", 3, 0);
        printf("SYN SENT\n");
        zmq_recv(ampOutIn_socket, ampOutInData, sample_size / 8, 0);
        printf("ACK SENT\n");
        zmq_send(ampOutIn_socket, "ACK", 3, 0);
        
        for (size_t i = 0; i < min((sample_size / 32), 16); ++i)
        {
            printf("0x%0004X: %s\n", ampOutInData[i], std::bitset<32>(ampOutInData[i]).to_string().c_str());
        }
    }

    zmq_unbind(ampOutIn_socket, privacyAmplificationServer_address);
    zmq_close(ampOutIn_socket);
    zmq_ctx_destroy(ampOutIn_socket);
    return 0;
}
