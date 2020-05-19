#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <zmq.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <thread>
#include <chrono>
#include <atomic>
#include <bitset>
#include <future>

typedef half    Real;
typedef half2   Complex;

#define factor 32
#define sample_size 256 //1024 * 1024 * factor
#define reduction 2048
#define min_template(a,b) (((a) < (b)) ? (a) : (b))
#define AMPOUT_REVERSE_ENDIAN TRUE
const Real normalisation_half = __float2half_rn((float)sample_size/(float)reduction/(float)reduction);

const char* address_seed_in = "tcp://127.0.0.1:45555"; //seed_in_alice
//const char* address_seed_in = "tcp://127.0.0.1:46666"; //seed_in_bob
const char* address_key_in = "tcp://127.0.0.1:47777"; //key_in
const char* address_amp_out = "tcp://127.0.0.1:48888"; //amp_out
constexpr int vertical_len = 96;
constexpr int horizontal_len = 160;
constexpr int key_len = 257;
constexpr int vertical_block = vertical_len / 32;
constexpr int horizontal_block = horizontal_len / 32;
constexpr int key_blocks = vertical_block + horizontal_block + 1;
constexpr int desired_block = vertical_block + horizontal_block;
constexpr int desired_len = vertical_len + horizontal_len;
unsigned int* toeplitz_seed = (unsigned int*)malloc(desired_block * 100);
unsigned int* key_start = (unsigned int*)malloc(desired_block * 100);
unsigned int* key_rest = (unsigned int*)malloc(desired_block * 100);
std::atomic<int> continueGeneratingNextBlock = 0;
std::atomic<int> blockReady = 0;
std::mutex printlock;
char syn[3];
char ack[3];


__device__ __constant__ half h0_dev;
__device__ __constant__ half h1_reduced_dev;
__device__ __constant__ half normalisation_half_dev;

__device__ __constant__ unsigned int intTobinMask_dev[32] =
{
    0b10000000000000000000000000000000,
    0b01000000000000000000000000000000,
    0b00100000000000000000000000000000,
    0b00010000000000000000000000000000,
    0b00001000000000000000000000000000,
    0b00000100000000000000000000000000,
    0b00000010000000000000000000000000,
    0b00000001000000000000000000000000,
    0b00000000100000000000000000000000,
    0b00000000010000000000000000000000,
    0b00000000001000000000000000000000,
    0b00000000000100000000000000000000,
    0b00000000000010000000000000000000,
    0b00000000000001000000000000000000,
    0b00000000000000100000000000000000,
    0b00000000000000010000000000000000,
    0b00000000000000001000000000000000,
    0b00000000000000000100000000000000,
    0b00000000000000000010000000000000,
    0b00000000000000000001000000000000,
    0b00000000000000000000100000000000,
    0b00000000000000000000010000000000,
    0b00000000000000000000001000000000,
    0b00000000000000000000000100000000,
    0b00000000000000000000000010000000,
    0b00000000000000000000000001000000,
    0b00000000000000000000000000100000,
    0b00000000000000000000000000010000,
    0b00000000000000000000000000001000,
    0b00000000000000000000000000000100,
    0b00000000000000000000000000000010,
    0b00000000000000000000000000000001
};

__global__
void ElementWiseProduct(int n, Complex* do1, Complex* do2, Complex* mul1)
{
    //Requires at least sm_53 as sm_52 and below don't support half maths.
    //Tegra/Jetson from Maxwell, Pascal, Volta, Turing and probably the upcomming Ampere
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    mul1[i].x = do1[i].x * do2[i].x - do1[i].y * do2[i].y;
    mul1[i].y = do1[i].x * do2[i].y + do1[i].y * do2[i].x;
}

__global__
void ToHalfArray(int n, unsigned int b, Real* halfOut, Real normalisation_half)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = i * 32;
    register const half half0 = __float2half_rn(0.0f);
    register const half half1 = __float2half_rn(1.0f / reduction);

    halfOut[j]    = (b & 0b10000000000000000000000000000000 > 0) ? half1 : half0;
    halfOut[j+1]  = (b & 0b01000000000000000000000000000000 > 0) ? half1 : half0;
    halfOut[j+2]  = (b & 0b00100000000000000000000000000000 > 0) ? half1 : half0;
    halfOut[j+3]  = (b & 0b00010000000000000000000000000000 > 0) ? half1 : half0;
    halfOut[j+4]  = (b & 0b00001000000000000000000000000000 > 0) ? half1 : half0;
    halfOut[j+5]  = (b & 0b00000100000000000000000000000000 > 0) ? half1 : half0;
    halfOut[j+6]  = (b & 0b00000010000000000000000000000000 > 0) ? half1 : half0;
    halfOut[j+7]  = (b & 0b00000001000000000000000000000000 > 0) ? half1 : half0;
    halfOut[j+8]  = (b & 0b00000000100000000000000000000000 > 0) ? half1 : half0;
    halfOut[j+9]  = (b & 0b00000000010000000000000000000000 > 0) ? half1 : half0;
    halfOut[j+10] = (b & 0b00000000001000000000000000000000 > 0) ? half1 : half0;
    halfOut[j+11] = (b & 0b00000000000100000000000000000000 > 0) ? half1 : half0;
    halfOut[j+12] = (b & 0b00000000000010000000000000000000 > 0) ? half1 : half0;
    halfOut[j+13] = (b & 0b00000000000001000000000000000000 > 0) ? half1 : half0;
    halfOut[j+14] = (b & 0b00000000000000100000000000000000 > 0) ? half1 : half0;
    halfOut[j+15] = (b & 0b00000000000000010000000000000000 > 0) ? half1 : half0;
    halfOut[j+16] = (b & 0b00000000000000001000000000000000 > 0) ? half1 : half0;
    halfOut[j+17] = (b & 0b00000000000000000100000000000000 > 0) ? half1 : half0;
    halfOut[j+18] = (b & 0b00000000000000000010000000000000 > 0) ? half1 : half0;
    halfOut[j+19] = (b & 0b00000000000000000001000000000000 > 0) ? half1 : half0;
    halfOut[j+20] = (b & 0b00000000000000000000100000000000 > 0) ? half1 : half0;
    halfOut[j+21] = (b & 0b00000000000000000000010000000000 > 0) ? half1 : half0;
    halfOut[j+22] = (b & 0b00000000000000000000001000000000 > 0) ? half1 : half0;
    halfOut[j+23] = (b & 0b00000000000000000000000100000000 > 0) ? half1 : half0;
    halfOut[j+24] = (b & 0b00000000000000000000000010000000 > 0) ? half1 : half0;
    halfOut[j+25] = (b & 0b00000000000000000000000001000000 > 0) ? half1 : half0;
    halfOut[j+26] = (b & 0b00000000000000000000000000100000 > 0) ? half1 : half0;
    halfOut[j+27] = (b & 0b00000000000000000000000000010000 > 0) ? half1 : half0;
    halfOut[j+28] = (b & 0b00000000000000000000000000001000 > 0) ? half1 : half0;
    halfOut[j+29] = (b & 0b00000000000000000000000000000100 > 0) ? half1 : half0;
    halfOut[j+30] = (b & 0b00000000000000000000000000000010 > 0) ? half1 : half0;
    halfOut[j+31] = (b & 0b00000000000000000000000000000001 > 0) ? half1 : half0;
}

__global__
void ToBinaryArray(Real* invOut, unsigned int* binOut, unsigned int* key_rest_dev)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = i * 32;
    binOut[i] = 
        (((__half2int_rn(invOut[j    ] / normalisation_half_dev) & 1) << 31) |
        ((__half2int_rn(invOut[j +  1] / normalisation_half_dev) & 1) << 30) |
        ((__half2int_rn(invOut[j +  2] / normalisation_half_dev) & 1) << 29) |
        ((__half2int_rn(invOut[j +  3] / normalisation_half_dev) & 1) << 28) |
        ((__half2int_rn(invOut[j +  4] / normalisation_half_dev) & 1) << 27) |
        ((__half2int_rn(invOut[j +  5] / normalisation_half_dev) & 1) << 26) |
        ((__half2int_rn(invOut[j +  6] / normalisation_half_dev) & 1) << 25) |
        ((__half2int_rn(invOut[j +  7] / normalisation_half_dev) & 1) << 24) |
        ((__half2int_rn(invOut[j +  8] / normalisation_half_dev) & 1) << 23) |
        ((__half2int_rn(invOut[j +  9] / normalisation_half_dev) & 1) << 22) |
        ((__half2int_rn(invOut[j + 10] / normalisation_half_dev) & 1) << 21) |
        ((__half2int_rn(invOut[j + 11] / normalisation_half_dev) & 1) << 20) |
        ((__half2int_rn(invOut[j + 12] / normalisation_half_dev) & 1) << 19) |
        ((__half2int_rn(invOut[j + 13] / normalisation_half_dev) & 1) << 18) |
        ((__half2int_rn(invOut[j + 14] / normalisation_half_dev) & 1) << 17) |
        ((__half2int_rn(invOut[j + 15] / normalisation_half_dev) & 1) << 16) |
        ((__half2int_rn(invOut[j + 16] / normalisation_half_dev) & 1) << 15) |
        ((__half2int_rn(invOut[j + 17] / normalisation_half_dev) & 1) << 14) |
        ((__half2int_rn(invOut[j + 18] / normalisation_half_dev) & 1) << 13) |
        ((__half2int_rn(invOut[j + 19] / normalisation_half_dev) & 1) << 12) |
        ((__half2int_rn(invOut[j + 20] / normalisation_half_dev) & 1) << 11) |
        ((__half2int_rn(invOut[j + 21] / normalisation_half_dev) & 1) << 10) |
        ((__half2int_rn(invOut[j + 22] / normalisation_half_dev) & 1) << 9) |
        ((__half2int_rn(invOut[j + 23] / normalisation_half_dev) & 1) << 8) |
        ((__half2int_rn(invOut[j + 24] / normalisation_half_dev) & 1) << 7) |
        ((__half2int_rn(invOut[j + 25] / normalisation_half_dev) & 1) << 6) |
        ((__half2int_rn(invOut[j + 26] / normalisation_half_dev) & 1) << 5) |
        ((__half2int_rn(invOut[j + 27] / normalisation_half_dev) & 1) << 4) |
        ((__half2int_rn(invOut[j + 28] / normalisation_half_dev) & 1) << 3) |
        ((__half2int_rn(invOut[j + 29] / normalisation_half_dev) & 1) << 2) |
        ((__half2int_rn(invOut[j + 30] / normalisation_half_dev) & 1) << 1) |
         (__half2int_rn(invOut[j + 31] / normalisation_half_dev) & 1)) ^ key_rest_dev[i];
}

__global__
void ToBinaryArray_reverse_endianness(Real* invOut, unsigned int* binOut, unsigned int* key_rest_dev)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int key_rest_little = key_rest_dev[i];
    int key_rest_big =
        (key_rest_little << 24) |
        ((key_rest_little << 8) & 0x00ff0000) |
        ((key_rest_little >> 8) & 0x0000ff00) |
        (key_rest_little >> 24);
    int j = i * 32;
    binOut[i] =
        (((__half2int_rn(invOut[j    ] / normalisation_half_dev) & 1) << 7) |
        ((__half2int_rn(invOut[j +  1] / normalisation_half_dev) & 1) << 6) |
        ((__half2int_rn(invOut[j +  2] / normalisation_half_dev) & 1) << 5) |
        ((__half2int_rn(invOut[j +  3] / normalisation_half_dev) & 1) << 4) |
        ((__half2int_rn(invOut[j +  4] / normalisation_half_dev) & 1) << 3) |
        ((__half2int_rn(invOut[j +  5] / normalisation_half_dev) & 1) << 2) |
        ((__half2int_rn(invOut[j +  6] / normalisation_half_dev) & 1) << 1) |
        ((__half2int_rn(invOut[j +  7] / normalisation_half_dev) & 1) << 0) |
        ((__half2int_rn(invOut[j +  8] / normalisation_half_dev) & 1) << 15) |
        ((__half2int_rn(invOut[j +  9] / normalisation_half_dev) & 1) << 14) |
        ((__half2int_rn(invOut[j + 10] / normalisation_half_dev) & 1) << 13) |
        ((__half2int_rn(invOut[j + 11] / normalisation_half_dev) & 1) << 12) |
        ((__half2int_rn(invOut[j + 12] / normalisation_half_dev) & 1) << 11) |
        ((__half2int_rn(invOut[j + 13] / normalisation_half_dev) & 1) << 10) |
        ((__half2int_rn(invOut[j + 14] / normalisation_half_dev) & 1) << 9) |
        ((__half2int_rn(invOut[j + 15] / normalisation_half_dev) & 1) << 8) |
        ((__half2int_rn(invOut[j + 16] / normalisation_half_dev) & 1) << 23) |
        ((__half2int_rn(invOut[j + 17] / normalisation_half_dev) & 1) << 22) |
        ((__half2int_rn(invOut[j + 18] / normalisation_half_dev) & 1) << 21) |
        ((__half2int_rn(invOut[j + 19] / normalisation_half_dev) & 1) << 20) |
        ((__half2int_rn(invOut[j + 20] / normalisation_half_dev) & 1) << 19) |
        ((__half2int_rn(invOut[j + 21] / normalisation_half_dev) & 1) << 18) |
        ((__half2int_rn(invOut[j + 22] / normalisation_half_dev) & 1) << 17) |
        ((__half2int_rn(invOut[j + 23] / normalisation_half_dev) & 1) << 16) |
        ((__half2int_rn(invOut[j + 24] / normalisation_half_dev) & 1) << 31) |
        ((__half2int_rn(invOut[j + 25] / normalisation_half_dev) & 1) << 30) |
        ((__half2int_rn(invOut[j + 26] / normalisation_half_dev) & 1) << 29) |
        ((__half2int_rn(invOut[j + 27] / normalisation_half_dev) & 1) << 28) |
        ((__half2int_rn(invOut[j + 28] / normalisation_half_dev) & 1) << 27) |
        ((__half2int_rn(invOut[j + 29] / normalisation_half_dev) & 1) << 26) |
        ((__half2int_rn(invOut[j + 30] / normalisation_half_dev) & 1) << 25) |
        ((__half2int_rn(invOut[j + 31] / normalisation_half_dev) & 1) << 24)) ^ key_rest_big;
}

__global__ void binInt2half(unsigned int* binIn, Real* realOut)
{
    unsigned int i;
    int block = blockIdx.x;
    int idx = threadIdx.x;
    unsigned int pos;
    unsigned int databyte;

    pos = (1024 * block * 32) + (idx * 32);
    databyte = binIn[1024 * block + idx];

    #pragma unroll (32)
    for (i = 0; i < 32; ++i)
    {
        if ((databyte & intTobinMask_dev[i]) == 0) {
            realOut[pos++] = h0_dev;
        }
        else
        {
            realOut[pos++] = h1_reduced_dev;
        }
    }
}

void intToBinCPU(int* intIn, unsigned int* binOut, int outSize) {
    int j = 0;
    for (int i = 0; i < outSize; ++i) {
        binOut[i] =
            (intIn[j] & 1 << 31) |
            (intIn[j + 1] & 1 << 30) |
            (intIn[j + 2] & 1 << 29) |
            (intIn[j + 3] & 1 << 28) |
            (intIn[j + 4] & 1 << 27) |
            (intIn[j + 5] & 1 << 26) |
            (intIn[j + 6] & 1 << 25) |
            (intIn[j + 7] & 1 << 24) |
            (intIn[j + 8] & 1 << 23) |
            (intIn[j + 9] & 1 << 22) |
            (intIn[j + 10] & 1 << 21) |
            (intIn[j + 11] & 1 << 20) |
            (intIn[j + 12] & 1 << 19) |
            (intIn[j + 13] & 1 << 18) |
            (intIn[j + 14] & 1 << 17) |
            (intIn[j + 15] & 1 << 16) |
            (intIn[j + 16] & 1 << 15) |
            (intIn[j + 17] & 1 << 14) |
            (intIn[j + 18] & 1 << 13) |
            (intIn[j + 19] & 1 << 12) |
            (intIn[j + 20] & 1 << 11) |
            (intIn[j + 21] & 1 << 10) |
            (intIn[j + 22] & 1 << 9) |
            (intIn[j + 23] & 1 << 8) |
            (intIn[j + 24] & 1 << 7) |
            (intIn[j + 25] & 1 << 6) |
            (intIn[j + 26] & 1 << 5) |
            (intIn[j + 27] & 1 << 4) |
            (intIn[j + 28] & 1 << 3) |
            (intIn[j + 29] & 1 << 2) |
            (intIn[j + 30] & 1 << 1) |
            (intIn[j + 31] & 1);
        j += 32;
    }
}


void printBin(const unsigned char* position, const unsigned char* end) {
    while (position < end) {
        printf("%s", std::bitset<8>(*position).to_string().c_str());
        ++position;
    }
    std::cout << std::endl;
}

void printBin(const unsigned int* position, const unsigned int* end) {
    while (position < end) {
        printf("%s", std::bitset<32>(*position).to_string().c_str());
        ++position;
    }
    std::cout << std::endl;
}






void recive() {
    void* context_seed_in = zmq_ctx_new();
    void* context_key_in = zmq_ctx_new();
    void* socket_seed_in = zmq_socket(context_seed_in, ZMQ_REQ);
    void* socket_key_in = zmq_socket(context_key_in, ZMQ_REQ);
    zmq_connect(socket_seed_in, address_seed_in);
    zmq_connect(socket_key_in, address_key_in);
    unsigned int recv_key[key_blocks];

    while (true) {
        printf("socket_seed_in\n");
        zmq_send(socket_seed_in, "SYN", 3, 0);
        printf("SYN SENT\n");
        zmq_recv(socket_seed_in, toeplitz_seed, desired_block * sizeof(unsigned int), 0);
        printf("ACK SENT\n");
        zmq_send(socket_seed_in, "ACK", 3, 0);
        printf("socket_key_in\n");
        zmq_send(socket_key_in, "SYN", 3, 0);
        zmq_recv(socket_key_in, recv_key, key_blocks * sizeof(unsigned int), 0);
        zmq_send(socket_key_in, "ACK", 3, 0);

        memcpy(key_start, recv_key, key_blocks * sizeof(unsigned int));
        *(key_start + horizontal_block) = *(recv_key + horizontal_block) & 0b10000000000000000000000000000000;
        memset(key_start + horizontal_block + 1, 0b00000000, (desired_block - horizontal_block - 1) * sizeof(unsigned int));

        int j = horizontal_block;
        for (int i = 0; i < vertical_block + 1; ++i)
        {
            key_rest[i] = ((recv_key[j] << 1) | (recv_key[j + 1] >> 31));
            ++j;
        }
        memset(key_rest + desired_block - horizontal_block, 0b00000000, vertical_block - (vertical_block - horizontal_block));

        printlock.lock();
        std::cout << "Toeplitz Seed: ";
        printBin(toeplitz_seed, toeplitz_seed + desired_block);
        std::cout << "Key: ";
        printBin(recv_key, recv_key + key_blocks);
        std::cout << "Key Start: ";
        printBin(key_start, key_start + desired_block + 1);
        std::cout << "Key Rest: ";
        printBin(key_rest, key_rest + vertical_block + 1);
        fflush(stdout);
        printlock.unlock();

        blockReady = 1;
        while (continueGeneratingNextBlock == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        continueGeneratingNextBlock = 0;
    }

    zmq_disconnect(socket_key_in, address_key_in);
    zmq_close(socket_key_in);
    zmq_ctx_destroy(socket_key_in);
}


int main(int argc, char* argv[])
{

    void* amp_out_context = zmq_ctx_new();
    void* amp_out_socket = zmq_socket(amp_out_context, ZMQ_REP);
    int rc = zmq_bind(amp_out_socket, address_amp_out);
    assert(rc == 0);

#ifdef _WIN32
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    DWORD dwConSize;
    COORD coordScreen = { 0, 0 };
    DWORD cCharsWritten;
    GetConsoleScreenBufferInfo(hConsole, &csbi);
    dwConSize = csbi.dwSize.X * csbi.dwSize.Y;
    FillConsoleOutputAttribute(hConsole, FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY | BACKGROUND_BLUE, dwConSize, coordScreen, &cCharsWritten);
    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY | BACKGROUND_BLUE);
#endif



    std::thread threadReciveObj(recive);
    threadReciveObj.detach();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    const int batch_size = 1; ; //Storage would also have to be increased for this to work
    long long int dist_sample = sample_size, dist_freq = sample_size / 2 + 1;
    const int loops = 10000;

    cufftHandle plan_forward_R2C_1, plan_forward_R2C_2, plan_inverse_C2R;
    unsigned int* key_start_dev;
    unsigned int* key_rest_dev;
    unsigned int* toeplitz_seed_dev;
    Real* di1;
    Real* di2;
    Real* invOut;
    Complex* do1;
    Complex* do2;
    Complex* mul1;
    unsigned int* binOut;
    unsigned char* Output;
    half* OutputHalf;
    cudaStream_t cpu2gpuStream1, cpu2gpuStream2, gpu2cpuStream, ElementWiseProductStream, ToBinaryArrayStream;
    cudaStreamCreate(&cpu2gpuStream1);
    cudaStreamCreate(&cpu2gpuStream2);
    cudaStreamCreate(&gpu2cpuStream);
    cudaStreamCreate(&ElementWiseProductStream);
    cudaStreamCreate(&ToBinaryArrayStream);

    // create cuda event to measure the performance
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate host pinned memory on RAM
    cudaMallocHost((void**)&Output, vertical_block * sizeof(unsigned int));
    cudaMallocHost((void**)&OutputHalf, sizeof(half*) * dist_sample);

    // Allocate memory on GPU
    cudaMalloc((void**)&key_start_dev, (sample_size/8));
    cudaMalloc((void**)&key_rest_dev, (sample_size/8));
    cudaMalloc((void**)&toeplitz_seed_dev, (sample_size/8));
    cudaMalloc((void**)&di1, sizeof(Real) * sample_size);
    cudaMalloc((void**)&di2, sizeof(Real) * sample_size);
    cudaMalloc((void**)&do1, sample_size * sizeof(Complex));
    cudaMalloc((void**)&do2, sample_size * sizeof(Complex));
    cudaMalloc(&mul1, sizeof(Complex) * dist_sample);
    cudaMalloc(&invOut, sizeof(Real) * dist_sample);
    cudaMalloc(&binOut, sizeof(unsigned int) * sample_size/8);

    register const half half0 = __float2half_rn(0.0f);
    register const half half1_reduced = __float2half_rn(1.0f/reduction);

    cudaMemcpyToSymbol(h0_dev, &half0, sizeof(half));
    cudaMemcpyToSymbol(h1_reduced_dev, &half1_reduced, sizeof(half));
    cudaMemcpyToSymbol(normalisation_half_dev, &normalisation_half, sizeof(half));


    int rank = 1;
    int stride_sample = 1, stride_freq = 1;
    long long embed_sample[] = { 0 };
    long long embedo1[] = { 0 };
    size_t workSize = 0;
    cufftCreate(&plan_forward_R2C_1);
    cufftXtMakePlanMany(plan_forward_R2C_1,
        rank, &dist_sample,
        embed_sample, stride_sample, dist_sample, CUDA_R_16F,
        embedo1, stride_freq, dist_freq, CUDA_C_16F,
        batch_size, &workSize, CUDA_C_16F);
    //cufftXtMakePlanMany(plan_forward_R2C_1, 1, &dist_sample, NULL, 1, 1,
    //    CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, 1, &workSize, CUDA_C_16F);
    cufftCreate(&plan_forward_R2C_2);
    cufftXtMakePlanMany(plan_forward_R2C_2,
        rank, &dist_sample,
        embed_sample, stride_sample, dist_sample, CUDA_R_16F,
        embedo1, stride_freq, dist_freq, CUDA_C_16F,
        batch_size, &workSize, CUDA_C_16F);
    cufftCreate(&plan_inverse_C2R);
    cufftXtMakePlanMany(plan_inverse_C2R,
        rank, &dist_sample,
        embedo1, stride_freq, dist_freq, CUDA_C_16F,
        embed_sample, stride_sample, dist_sample, CUDA_R_16F,
        batch_size, &workSize, CUDA_R_16F);

    while (true) {
        
        register const half half0 = __float2half_rn(0.0f);
        register const half half1 = __float2half_rn(1.0f / reduction);

        printlock.lock();
        printf("Bob!!!\n");
        fflush(stdout);
        printlock.unlock();
        while (blockReady == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        printlock.lock();
        printf("Ready!!!\n");
        fflush(stdout);
        printlock.unlock();
        
        cudaMemcpy(key_start_dev, key_start, dist_sample / 8, cudaMemcpyHostToDevice);
        cudaMemcpy(key_rest_dev, key_rest, dist_sample / 8, cudaMemcpyHostToDevice);
        cudaMemset(key_rest_dev + horizontal_block + 1, 0, (sample_size / 8) - horizontal_block + 1);
        cudaMemcpy(toeplitz_seed_dev, toeplitz_seed, dist_sample / 8, cudaMemcpyHostToDevice);
        
        blockReady = 0;
        continueGeneratingNextBlock = 1;
        binInt2half <<<(int)(((int)(sample_size / 32) + 1023) / 1024), std::min(sample_size / 32, 1024), 0,
            ToBinaryArrayStream >>> (key_start_dev, di1);
        binInt2half <<<(int)(((int)(sample_size / 32) + 1023) / 1024), std::min(sample_size / 32, 1024), 0,
            ToBinaryArrayStream >>> (toeplitz_seed_dev, di2);
        
        cudaStreamSynchronize(ToBinaryArrayStream);


        cudaEventRecord(start);
        cudaEventSynchronize(start);
        //cudaMemcpyAsync(di1, hi1, dist_sample * sizeof(Real), cudaMemcpyHostToDevice, cpu2gpuStream1);
        //for (int i = 0; i < loops/factor; ++i) {
        //cudaMemcpyAsync(di2, hi2, dist_sample * sizeof(Real), cudaMemcpyHostToDevice, cpu2gpuStream2);
        //cudaStreamSynchronize(cpu2gpuStream1);
        cufftXtExec(plan_forward_R2C_1, di1, do1, CUFFT_FORWARD);
        //cudaStreamSynchronize(cpu2gpuStream2);
        cufftXtExec(plan_forward_R2C_2, di2, do2, CUFFT_FORWARD);
        ElementWiseProduct <<<(int)((dist_freq + 1023) / 1024), std::min((int)dist_freq, 1024), 0, ElementWiseProductStream >>> (dist_freq, do1, do2, mul1);
        cudaStreamSynchronize(ElementWiseProductStream);
        cufftXtExec(plan_inverse_C2R, mul1, invOut, CUFFT_INVERSE);
        #if AMPOUT_REVERSE_ENDIAN == TRUE
        ToBinaryArray_reverse_endianness <<<(int)(((int)(vertical_block) + 1023) / 1024), std::min(vertical_block, 1024), 0, ToBinaryArrayStream >>> (invOut, binOut, key_rest_dev);
        #else
        ToBinaryArray << <(int)(((int)(vertical_block)+1023) / 1024), std::min(vertical_block, 1024), 0, ToBinaryArrayStream >> > (invOut, binOut, key_rest_dev);
        #endif
        cudaStreamSynchronize(ToBinaryArrayStream);
        cudaMemcpy(Output, binOut, vertical_block * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        //}
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float et;
        cudaEventElapsedTime(&et, start, stop);
        //printf("FFT time for %lld samples: %f ms\n", (sample_size * loops), et);

        printlock.lock();

        zmq_recv(amp_out_socket, syn, 3, 0);
        printf("Recived: %c%c%c\n", syn[0], syn[1], syn[2]);
        zmq_send(amp_out_socket, Output, sample_size / 8, 0);
        zmq_recv(amp_out_socket, ack, 3, 0);
        printf("Recived: %c%c%c\n", ack[0], ack[1], ack[2]);
        
        for (size_t i = 0; i < min_template(vertical_block * sizeof(unsigned int), 16); ++i)
        {
            printf("0x%02X: %s\n", Output[i], std::bitset<8>(Output[i]).to_string().c_str());
        }
        fflush(stdout);
        printlock.unlock();

    }


    // Delete CUFFT Plans
    cufftDestroy(plan_forward_R2C_1);
    cufftDestroy(plan_forward_R2C_2);
    cufftDestroy(plan_inverse_C2R);

    // Deallocate memoriey on GPU and RAM
    cudaFree(di1);
    cudaFree(di2);
    cudaFree(invOut);
    cudaFree(do1);
    cudaFree(do2);
    cudaFree(mul1);
    cudaFree(binOut);
    cudaFree(Output);

    // Delete cuda events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
