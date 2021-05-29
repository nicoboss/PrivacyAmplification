#define _CRT_SECURE_NO_WARNINGS
#include <iomanip>
#include <iostream>
#include <time.h>
#include <mutex>
#include <bitset>
#include <zmq.h>
using namespace std;

/*Privacy Amplification input size in bits
  Has to be 2^x and 2^27 is the maximum
  Needs to match with the one specified in other components*/
#define factor 11
#define pwrtwo(x) (1 << (x))
#define sample_size pwrtwo(factor)

/*Address of the Privacy Amplification Server*/
const char* privacyAmplificationServer_address = "tcp://127.0.0.1:48888";
constexpr uint32_t vertical_len = sample_size / 4 + sample_size / 8;
constexpr uint32_t vertical_bytes = vertical_len / 8;
#define min(a,b) (((a) < (b)) ? (a) : (b))
unsigned char* ampOutInData = (unsigned char*)malloc(vertical_bytes);


/// @brief Receives data from the Privacy Amplification Sevrer
/// Contains all the comunication and error handling required
/// to receive the privacy amplification results from the Privacy Amplification Server.
/// Currently it prints the first 4 bytes of the result on the screen.
/// Note: In a real environment this code should be integrated in whatever
/// tool makes use of the Privacy Amplification result.
int main(int argc, char* argv[])
{
	int32_t rc;
	time_t currentTime;
	void* context = zmq_ctx_new();

	reconnect:;
	void* ampOutIn_socket = zmq_socket(context, ZMQ_PULL);
	int hwm = 1;
	zmq_setsockopt(ampOutIn_socket, ZMQ_RCVHWM, &hwm, sizeof(int));

	cout << "Waiting for PrivacyAmplification Server..." << endl;
	zmq_connect(ampOutIn_socket, privacyAmplificationServer_address);

	while (true) {
		rc = zmq_recv(ampOutIn_socket, ampOutInData, vertical_bytes, 0);
		if (rc != vertical_bytes) {
			cout << "Error receiving data from PrivacyAmplification Server!" << endl;
			cout << "Expected " << vertical_bytes << " bytes but received " << rc << " bytes! Retrying..." << endl;
			zmq_close(ampOutIn_socket);
			goto reconnect;
		}

		time(&currentTime);
		cout << put_time(localtime(&currentTime), "%F %T") << " Key Block recived" << endl;

		//for (size_t i = 0; i < min(vertical_bytes, 4); ++i)
		//{
		//	printf("0x%02X: %s\n", ampOutInData[i], bitset<8>(ampOutInData[i]).to_string().c_str());
		//}
	}

	zmq_close(ampOutIn_socket);
	zmq_ctx_destroy(ampOutIn_socket);
	return 0;
}
