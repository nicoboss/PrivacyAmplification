#define _CRT_SECURE_NO_WARNINGS
#include <iomanip>
#include <iostream>
#include <time.h>
#include <mutex>
#include <bitset>
#include <zmq.h>

#define factor 27
#define pwrtwo(x) (1 << (x))
#define sample_size pwrtwo(factor)
const char* privacyAmplificationServer_address = "tcp://127.0.0.1:48888";
constexpr uint32_t vertical_len = sample_size / 4 + sample_size / 8;
constexpr uint32_t vertical_bytes = vertical_len / 8;
#define min(a,b) (((a) < (b)) ? (a) : (b))
unsigned char* ampOutInData = (unsigned char*)malloc(vertical_bytes);

int main(int argc, char* argv[])
{
    int32_t rc;
    time_t currentTime;
    void* context = zmq_ctx_new();
    int timeout = 1000;

    reconnect:;
    void* ampOutIn_socket = zmq_socket(context, ZMQ_REQ);
    zmq_setsockopt(ampOutIn_socket, ZMQ_RCVTIMEO, &timeout, sizeof(int));
    
    std::cout << "Waiting for PrivacyAmplification Server..." << std::endl;
    zmq_connect(ampOutIn_socket, privacyAmplificationServer_address);
    
    while (true) {
        zmq_send(ampOutIn_socket, "SYN", 3, 0);
        rc = zmq_recv(ampOutIn_socket, ampOutInData, vertical_bytes, 0);
        if (rc != vertical_bytes) {
            std::cout << "Error receiving data from PrivacyAmplification Server!" << std::endl;
            std::cout << "Expected " << vertical_bytes << " bytes but received " << rc << " bytes! Retrying..." << std::endl;
            zmq_close(ampOutIn_socket);
            goto reconnect;
        }

        time(&currentTime);
        std::cout << std::put_time(localtime(&currentTime), "%F %T") << " Key Block recived" << std::endl;
        
        for (size_t i = 0; i < min(vertical_bytes, 16); ++i)
        {
            printf("0x%02X: %s\n", ampOutInData[i], std::bitset<8>(ampOutInData[i]).to_string().c_str());
        }
    }

    zmq_close(ampOutIn_socket);
    zmq_ctx_destroy(ampOutIn_socket);
    return 0;
}
