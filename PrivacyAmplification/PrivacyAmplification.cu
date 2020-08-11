#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <zmq.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <thread>
#include <atomic>
#include <bitset>
#include <future>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

typedef float    Real;
typedef float2   Complex;

#define TRUE 1
#define FALSE 0
#define factor 27
#define pwrtwo(x) (1 << (x))
#define sample_size pwrtwo(factor)
#define reduction pwrtwo(11)
#define pre_mul_reduction pwrtwo(5)
#define total_reduction reduction*pre_mul_reduction
#define min_template(a,b) (((a) < (b)) ? (a) : (b))
#define INPUT_BLOCKS_TO_CACHE 16 //Has to be larger then 1
#define OUTPUT_BLOCKS_TO_CACHE 16 //Has to be larger then 1
#define DYNAMIC_TOEPLITZ_MATRIX_SEED TRUE
#define XOR_WITH_KEY_REST TRUE //Must be enabled for security
#define SHOW_AMPOUT TRUE
#define SHOW_DEBUG_OUTPUT FALSE
#define SHOW_SHOW_KEY_DEBUG_OUTPUT FALSE
#define USE_MATRIX_SEED_SERVER FALSE
#define USE_KEY_SERVER FALSE
#define HOST_AMPOUT_SERVER FALSE
#define STORE_FIRST_AMPOUT_IN_FILE TRUE
#define AMPOUT_REVERSE_ENDIAN TRUE
#define TOEPLITZ_SEED_PATH "toeplitz_seed.bin"
#define KEYFILE_PATH "keyfile.bin"
#define VERIFY_AMPOUT TRUE
#define print(TEXT) printStream(std::ostringstream().flush() << TEXT);
#define println(TEXT) printlnStream(std::ostringstream().flush() << TEXT);
#define minValue(a,b) (((a) < (b)) ? (a) : (b))
#define cudaCalloc(a,b) if (cudaMalloc(a, b) == cudaSuccess) cudaMemset(*a, 0b00000000, b);
const Real normalisation_float = ((float)sample_size)/((float)total_reduction)/((float)total_reduction);

#if VERIFY_AMPOUT == TRUE
#include "sha3.h"
const uint8_t ampout_sha3[] = { 0xC4, 0x22, 0xB6, 0x86, 0x5C, 0x72, 0xCA, 0xD8,
                               0x2C, 0xC2, 0x6A, 0x14, 0x62, 0xB8, 0xA4, 0x56,
                               0x6F, 0x91, 0x17, 0x50, 0xF3, 0x1B, 0x14, 0x75,
                               0x69, 0x12, 0x69, 0xC1, 0xB7, 0xD4, 0xA7, 0x16 };
#endif

#ifdef __CUDACC__
#define KERNEL_ARG2(grid, block) <<< grid, block >>>
#define KERNEL_ARG3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARG2(grid, block)
#define KERNEL_ARG3(grid, block, sh_mem)
#define KERNEL_ARG4(grid, block, sh_mem, stream)
#endif

#ifdef __INTELLISENSE__
cudaError_t cudaMemcpyToSymbol(Complex symbol, const void* src, size_t count);
cudaError_t cudaMemcpyToSymbol(Real symbol, const void* src, size_t count);
int __float2int_rn(float in);
unsigned int atomicAdd(unsigned int* address, unsigned int val);
#define __syncthreads()
#endif

#if USE_MATRIX_SEED_SERVER == TRUE
const char* address_seed_in = "tcp://127.0.0.1:45555"; //seed_in_alice
//const char* address_seed_in = "tcp://127.0.0.1:46666"; //seed_in_bob
#endif
#if USE_KEY_SERVER == TRUE
const char* address_key_in = "tcp://127.0.0.1:47777"; //key_in
#endif
#if HOST_AMPOUT_SERVER == TRUE
const char* address_amp_out = "tcp://127.0.0.1:48888"; //amp_out
#endif
uint32_t vertical_len = sample_size/4 + sample_size/8;
uint32_t horizontal_len = sample_size/2 + sample_size/8;
uint32_t vertical_block = vertical_len / 32;
uint32_t horizontal_block = horizontal_len / 32;
constexpr uint32_t desired_block = sample_size / 32;
constexpr uint32_t key_blocks = desired_block + 1;
constexpr uint32_t input_cache_block_size = desired_block;
constexpr uint32_t output_cache_block_size = (desired_block + 31) * sizeof(uint32_t);
uint32_t* recv_key = (uint32_t*)malloc(key_blocks * sizeof(uint32_t));
uint32_t* toeplitz_seed;
uint32_t* key_start;
uint32_t* key_start_zero_pos = (uint32_t*)malloc(INPUT_BLOCKS_TO_CACHE * sizeof(uint32_t));
uint32_t* key_rest;
uint32_t* key_rest_zero_pos = (uint32_t*)malloc(INPUT_BLOCKS_TO_CACHE * sizeof(uint32_t));
uint8_t * Output;
#if SHOW_DEBUG_OUTPUT == TRUE
Real* OutputFloat;
#endif
std::atomic<uint32_t> input_cache_read_pos;
std::atomic<uint32_t> input_cache_write_pos;
std::atomic<uint32_t> output_cache_read_pos;
std::atomic<uint32_t> output_cache_write_pos;
std::mutex printlock;


__device__ __constant__ Complex c0_dev;
__device__ __constant__ Real h0_dev;
__device__ __constant__ Real h1_reduced_dev;
__device__ __constant__ Real normalisation_float_dev;

__device__ __constant__ uint32_t intTobinMask_dev[32] =
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

__device__ __constant__ uint32_t ToBinaryBitShiftArray_dev[32] =
{
    #if AMPOUT_REVERSE_ENDIAN == TRUE
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24
    #else
    31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    #endif
};

void printStream(std::ostream& os) {
    std::ostringstream& ss = dynamic_cast<std::ostringstream&>(os);
    printlock.lock();
    std::cout << ss.str() << std::flush;
    printlock.unlock();
}

void printlnStream(std::ostream& os) {
    std::ostringstream& ss = dynamic_cast<std::ostringstream&>(os);
    printlock.lock();
    std::cout << ss.str() << std::endl;
    printlock.unlock();
}

__global__
void calculateCorrectionFloat(uint32_t* count_one_global_seed, uint32_t* count_one_global_key, float* correction_float_dev)
{
    //*correction_float_dev = (float)((unsigned long)(*count_one_global_key-60000000));
    uint64_t count_multiblicated = *count_one_global_seed * *count_one_global_key;
    double count_multiblicated_normalized = count_multiblicated / (double)sample_size;
    double two = 2.0;
    Real count_multiblicated_normalized_modulo = (float)modf(count_multiblicated_normalized, &two);
    *correction_float_dev = count_multiblicated_normalized_modulo;
}

__global__
void setFirstElementToZero(Complex* do1, Complex* do2)
{
    if (threadIdx.x == 0) {
        do1[0] = c0_dev;
    }
    else
    {
        do2[0] = c0_dev;
    }
}

__global__
void ElementWiseProduct(Complex* do1, Complex* do2)
{
    //Requires at least sm_53 as sm_52 and below don't support float maths.
    //Tegra/Jetson from Maxwell, Pascal, Volta, Turing and probably the upcomming Ampere
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float r = pre_mul_reduction;
    Real do1x = do1[i].x/r;
    Real do1y = do1[i].y/r;
    Real do2x = do2[i].x/r;
    Real do2y = do2[i].y/r;
    do1[i].x = do1x * do2x - do1y * do2y;
    do1[i].y = do1x * do2y + do1y * do2x;
}

__global__
void ToFloatArray(uint32_t n, uint32_t b, Real* floatOut, Real normalisation_float)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = i * 32;

    floatOut[j]    = (b & 0b10000000000000000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+1]  = (b & 0b01000000000000000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+2]  = (b & 0b00100000000000000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+3]  = (b & 0b00010000000000000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+4]  = (b & 0b00001000000000000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+5]  = (b & 0b00000100000000000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+6]  = (b & 0b00000010000000000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+7]  = (b & 0b00000001000000000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+8]  = (b & 0b00000000100000000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+9]  = (b & 0b00000000010000000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+10] = (b & 0b00000000001000000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+11] = (b & 0b00000000000100000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+12] = (b & 0b00000000000010000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+13] = (b & 0b00000000000001000000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+14] = (b & 0b00000000000000100000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+15] = (b & 0b00000000000000010000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+16] = (b & 0b00000000000000001000000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+17] = (b & 0b00000000000000000100000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+18] = (b & 0b00000000000000000010000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+19] = (b & 0b00000000000000000001000000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+20] = (b & 0b00000000000000000000100000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+21] = (b & 0b00000000000000000000010000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+22] = (b & 0b00000000000000000000001000000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+23] = (b & 0b00000000000000000000000100000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+24] = (b & 0b00000000000000000000000010000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+25] = (b & 0b00000000000000000000000001000000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+26] = (b & 0b00000000000000000000000000100000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+27] = (b & 0b00000000000000000000000000010000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+28] = (b & 0b00000000000000000000000000001000 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+29] = (b & 0b00000000000000000000000000000100 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+30] = (b & 0b00000000000000000000000000000010 > 0) ? h1_reduced_dev : h0_dev;
    floatOut[j+31] = (b & 0b00000000000000000000000000000001 > 0) ? h1_reduced_dev : h0_dev;
}

__global__
void ToBinaryArray(Real* invOut, uint32_t* binOut, uint32_t* key_rest_local, Real* correction_float_dev)
{
    const Real normalisation_float_local = normalisation_float_dev;
    const uint32_t block = blockIdx.x;
    const uint32_t idx = threadIdx.x;
    const Real correction_float = *correction_float_dev;
    
    __shared__ uint32_t key_rest_xor[31];
    __shared__ uint32_t binOutRawBit[992];
    if (idx < 992) {
        binOutRawBit[idx] = ((__float2int_rn(invOut[block * 992 + idx] / normalisation_float_local + correction_float) & 1) << ToBinaryBitShiftArray_dev[idx % 32]);
    }
    else if (idx < 1023)
    {
        #if AMPOUT_REVERSE_ENDIAN == TRUE
        uint32_t key_rest_little = key_rest_local[block * 31 + idx - 992];
        key_rest_xor[idx - 992] =
            ((((key_rest_little) & 0xff000000) >> 24) |
                (((key_rest_little) & 0x00ff0000) >> 8) |
                (((key_rest_little) & 0x0000ff00) << 8) |
                (((key_rest_little) & 0x000000ff) << 24));
        #else
                key_rest_xor[idx - 992] = key_rest_local[block * 31 + idx - 992];
        #endif 
    }
    __syncthreads();

    if (idx < 31) {
        const uint32_t pos = idx * 32;
        uint32_t binOutLocal =
            (binOutRawBit[pos] | binOutRawBit[pos + 1] | binOutRawBit[pos + 2] | binOutRawBit[pos + 3] |
            binOutRawBit[pos + 4] | binOutRawBit[pos + 5] | binOutRawBit[pos + 6] | binOutRawBit[pos + 7] |
            binOutRawBit[pos + 8] | binOutRawBit[pos + 9] | binOutRawBit[pos + 10] | binOutRawBit[pos + 11] |
            binOutRawBit[pos + 12] | binOutRawBit[pos + 13] | binOutRawBit[pos + 14] | binOutRawBit[pos + 15] |
            binOutRawBit[pos + 16] | binOutRawBit[pos + 17] | binOutRawBit[pos + 18] | binOutRawBit[pos + 19] |
            binOutRawBit[pos + 20] | binOutRawBit[pos + 21] | binOutRawBit[pos + 22] | binOutRawBit[pos + 23] |
            binOutRawBit[pos + 24] | binOutRawBit[pos + 25] | binOutRawBit[pos + 26] | binOutRawBit[pos + 27] |
            binOutRawBit[pos + 28] | binOutRawBit[pos + 29] | binOutRawBit[pos + 30] | binOutRawBit[pos + 31])
            #if XOR_WITH_KEY_REST == TRUE 
            ^ key_rest_xor[idx]
            #endif
            ;
        binOut[block * 31 + idx] = binOutLocal;
    }
}

__global__
void binInt2float(uint32_t* binIn, Real* realOut, uint32_t* count_one_global)
{
    //Multicast
    Real h0_local = h0_dev;
    Real h1_reduced_local = h1_reduced_dev;
    __shared__ uint32_t binInShared[32];

    uint32_t block = blockIdx.x;
    uint32_t idx = threadIdx.x;
    uint32_t maskToUse;
    uint32_t inPos;
    uint32_t outPos;
    maskToUse = idx % 32;
    inPos = idx / 32;
    outPos = 1024 * block + idx;

    if (threadIdx.x < 32) {
        binInShared[idx] = binIn[32 * block + idx];
    }
    __syncthreads();

    if ((binInShared[inPos] & intTobinMask_dev[maskToUse]) == 0) {
        realOut[outPos] = h0_local;
    }
    else
    {
        atomicAdd(count_one_global, 1);
        realOut[outPos] = h1_reduced_local;
    }
}

void intToBinCPU(int* intIn, uint32_t* binOut, uint32_t outSize) {
    uint32_t j = 0;
    for (uint32_t i = 0; i < outSize; ++i) {
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


void printBin(const uint8_t * position, const uint8_t * end) {
    while (position < end) {
        printf("%s", std::bitset<8>(*position).to_string().c_str());
        ++position;
    }
    std::cout << std::endl;
}

void printBin(const uint32_t* position, const uint32_t* end) {
    while (position < end) {
        printf("%s", std::bitset<32>(*position).to_string().c_str());
        ++position;
    }
    std::cout << std::endl;
}

inline void key2StartRest() {
    uint32_t* key_start_block = key_start + input_cache_block_size * input_cache_write_pos;
    uint32_t* key_rest_block = key_rest + input_cache_block_size * input_cache_write_pos;
    uint32_t* key_start_zero_pos_block = key_start_zero_pos + input_cache_write_pos;
    uint32_t* key_rest_zero_pos_block = key_rest_zero_pos + input_cache_write_pos;

    memcpy(key_start_block, recv_key, horizontal_block * sizeof(uint32_t));
    *(key_start_block + horizontal_block) = *(recv_key + horizontal_block) & 0b10000000000000000000000000000000;

    uint32_t j = horizontal_block;
    for (uint32_t i = 0; i < vertical_block - 1; ++i)
    {
        key_rest_block[i] = ((recv_key[j] << 1) | (recv_key[j + 1] >> 31));
        ++j;
    }
    key_rest_block[vertical_block - 1] = ((recv_key[j] << 1));

    uint32_t new_key_start_zero_pos = horizontal_block + 1;
    if (new_key_start_zero_pos < *key_start_zero_pos_block)
    {
        uint32_t key_start_fill_length = *key_start_zero_pos_block - new_key_start_zero_pos;
        memset(key_start_block + new_key_start_zero_pos, 0b00000000, key_start_fill_length * sizeof(uint32_t));
        *key_start_zero_pos_block = new_key_start_zero_pos;
    }
    
    uint32_t new_key_rest_zero_pos = desired_block - horizontal_block;
    if (new_key_rest_zero_pos < *key_rest_zero_pos_block)
    {
        uint32_t key_rest_fill_length = *key_rest_zero_pos_block - new_key_rest_zero_pos;
        memset(key_rest_block + new_key_rest_zero_pos, 0b00000000, key_rest_fill_length * sizeof(uint32_t));
        *key_rest_zero_pos_block = new_key_rest_zero_pos;
    }
}


void reciveData() {

    int32_t rc;
    #if USE_MATRIX_SEED_SERVER == TRUE
    void* context_seed_in = zmq_ctx_new();
    void* socket_seed_in = zmq_socket(context_seed_in, ZMQ_REQ);
    while(zmq_connect(socket_seed_in, address_seed_in) != 0) {
        println("Connection to \"" << address_seed_in << "\" failed! Retrying...");
    }
    #else
    //Cryptographically random Toeplitz seed generated by XOR a self-generated
    //VeraCrypt key file (PRF: SHA-512) with ANU_20Oct2017_100MB_7
    //from the ANU Quantum Random Numbers Server (https://qrng.anu.edu.au/)
    std::ifstream seedfile(TOEPLITZ_SEED_PATH, std::ios::binary);

    if (seedfile.fail())
    {
        std::cout << "Can't open file \"" << TOEPLITZ_SEED_PATH << "\" => terminating!" << std::endl;
        exit(1);
        abort();
    }

    seedfile.seekg(0, std::ios::end);
    size_t seedfile_length = seedfile.tellg();
    seedfile.seekg(0, std::ios::beg);

    if (seedfile_length < desired_block * sizeof(uint32_t))
    {
        std::cout << "File \"" << TOEPLITZ_SEED_PATH << "\" is with " << seedfile_length << " bytes too short!" << std::endl;
        std::cout << "it is required to be at least " << desired_block * sizeof(uint32_t) << " bytes => terminating!" << std::endl;
        exit(1);
        abort();
    }

    char* toeplitz_seed_char = reinterpret_cast<char*>(toeplitz_seed + input_cache_block_size * input_cache_write_pos);
    seedfile.read(toeplitz_seed_char, desired_block * sizeof(uint32_t));
    for (uint32_t i = 0; i < INPUT_BLOCKS_TO_CACHE; ++i) {
        uint32_t* toeplitz_seed_block = toeplitz_seed + input_cache_block_size * i;
        memcpy(toeplitz_seed_block, toeplitz_seed, input_cache_block_size * sizeof(uint32_t));
    }
    #endif

    #if USE_KEY_SERVER == TRUE
    void* context_key_in = zmq_ctx_new();
    void* socket_key_in = zmq_socket(context_key_in, ZMQ_REQ);
    while (zmq_connect(socket_key_in, address_key_in) != 0) {
        printlock.lock();
        std::cout << "Connection to \"" << address_key_in << "\" failed! Retrying..." << std::endl;
        printlock.unlock();
    }
    #else
    //Cryptographically random Toeplitz seed generated by XOR a self-generated
    //VeraCrypt key file (PRF: SHA-512) with ANU_20Oct2017_100MB_49
    //from the ANU Quantum Random Numbers Server (https://qrng.anu.edu.au/)
    std::ifstream keyfile(KEYFILE_PATH, std::ios::binary);

    if (keyfile.fail())
    {
        std::cout << "Can't open file \"" << KEYFILE_PATH << "\" => terminating!" << std::endl;
        exit(1);
        abort();
    }

    keyfile.seekg(0, std::ios::end);
    size_t keyfile_length = keyfile.tellg();
    keyfile.seekg(0, std::ios::beg);

    if (keyfile_length < key_blocks * sizeof(uint32_t))
    {
        std::cout << "File \"" << KEYFILE_PATH << "\" is with " << keyfile_length << " bytes too short!" << std::endl;
        std::cout << "it is required to be at least " << key_blocks * sizeof(uint32_t) << " bytes => terminating!" << std::endl;
        exit(1);
        abort();
    }

    char* recv_key_char = reinterpret_cast<char*>(recv_key);
    keyfile.read(recv_key_char, key_blocks * sizeof(uint32_t));
    key2StartRest();
    for (uint32_t i = 0; i < INPUT_BLOCKS_TO_CACHE; ++i) {
        uint32_t* key_start_block = key_start + input_cache_block_size * i;
        uint32_t* key_rest_block = key_rest + input_cache_block_size * i;
        uint32_t* key_start_zero_pos_block = key_start_zero_pos + i;
        uint32_t* key_rest_zero_pos_block = key_rest_zero_pos + i;
        memcpy(key_start_block, key_start, input_cache_block_size * sizeof(uint32_t));
        memcpy(key_rest_block, key_rest, input_cache_block_size * sizeof(uint32_t));
        *key_start_zero_pos_block = *key_start_zero_pos;
        *key_rest_zero_pos_block = *key_rest_zero_pos;
    }
    #endif

    bool recive_toeplitz_matrix_seed = true;
    while (true)
    {

        while (input_cache_write_pos % INPUT_BLOCKS_TO_CACHE == input_cache_read_pos) {
            std::this_thread::yield();
        }

        uint32_t* toeplitz_seed_block = toeplitz_seed + input_cache_block_size * input_cache_write_pos;
        #if USE_MATRIX_SEED_SERVER == TRUE
        if (recive_toeplitz_matrix_seed) {
            retry_receiving_seed:
            int32_t rc = zmq_send(socket_seed_in, "SYN", 3, 0);
            if (rc != 3) {
                println("Error sending SYN to Seedserver! Error Code: " << zmq_errno() << " - Retrying...");
                goto retry_receiving_seed;
            }
            if (zmq_recv(socket_seed_in, toeplitz_seed_block, desired_block * sizeof(uint32_t), 0) != desired_block * sizeof(uint32_t)) {
                println("Error receiving data from Seedserver! Retrying...");
                goto retry_receiving_seed;
            }
            println("Seed Block recived");

            #if DYNAMIC_TOEPLITZ_MATRIX_SEED == FALSE
            recive_toeplitz_matrix_seed = false;
            zmq_disconnect(socket_seed_in, address_seed_in);
            zmq_close(socket_seed_in);
            zmq_ctx_destroy(socket_seed_in);
            for (uint32_t i = 0; i < INPUT_BLOCKS_TO_CACHE; ++i) {
                uint32_t* toeplitz_seed_block = toeplitz_seed + input_cache_block_size * i;
                memcpy(toeplitz_seed_block, toeplitz_seed, input_cache_block_size * sizeof(uint32_t));
            }
            #endif
        }
        #endif

        #if USE_KEY_SERVER == TRUE
        retry_receiving_key:
        if (zmq_send(socket_key_in, "SYN", 3, 0) != 3) {
            println("Error sending SYN to Keyserver! Retrying...");
            goto retry_receiving_key;
        }
        if (zmq_recv(socket_key_in, &vertical_block, sizeof(uint32_t), 0) != sizeof(uint32_t)) {
            println("Error receiving vertical_blocks from Keyserver! Retrying...");
            goto retry_receiving_key;
        }
        vertical_len = vertical_block * 32;
        horizontal_len = sample_size - vertical_len;
        horizontal_block = horizontal_len / 32;
        if (zmq_recv(socket_key_in, recv_key, key_blocks * sizeof(uint32_t), 0) != key_blocks * sizeof(uint32_t)) {
            println("Error receiving data from Keyserver! Retrying...");
            goto retry_receiving_key;
        }
        println("Key Block recived");
        key2StartRest();
        #endif

        #if SHOW_KEY_DEBUG_OUTPUT == TRUE
        uint32_t* key_start_block = key_start + input_cache_block_size * input_cache_write_pos;
        uint32_t* key_rest_block = key_rest + input_cache_block_size * input_cache_write_pos;
        printlock.lock();
        std::cout << "Toeplitz Seed: ";
        printBin(toeplitz_seed_block, toeplitz_seed_block + desired_block);
        std::cout << "Key: ";
        printBin(recv_key, recv_key + key_blocks);
        std::cout << "Key Start: ";
        printBin(key_start_block, key_start_block + desired_block + 1);
        std::cout << "Key Rest: ";
        printBin(key_rest_block, key_rest_block + vertical_block + 1);
        fflush(stdout);
        printlock.unlock();
        #endif

        input_cache_write_pos = (input_cache_write_pos + 1) % INPUT_BLOCKS_TO_CACHE;

    }

    #if USE_MATRIX_SEED_SERVER == TRUE
    if (recive_toeplitz_matrix_seed) {
        zmq_disconnect(socket_seed_in, address_seed_in);
        zmq_close(socket_seed_in);
        zmq_ctx_destroy(socket_seed_in);
    }
    #endif

    #if USE_KEY_SERVER == TRUE
    zmq_disconnect(socket_key_in, address_key_in);
    zmq_close(socket_key_in);
    zmq_ctx_destroy(socket_key_in);
    #endif
}


void sendData() {
    #if HOST_AMPOUT_SERVER == TRUE
    char syn[3];
    int32_t rc;
    void* amp_out_context = zmq_ctx_new();
    void* amp_out_socket = zmq_socket(amp_out_context, ZMQ_REP);
    while (zmq_bind(amp_out_socket, address_amp_out) != 0) {
        println("Binding to \"" << address_amp_out << "\" failed! Retrying...");
    }
    #endif
    #if STORE_FIRST_AMPOUT_IN_FILE == TRUE
    bool firstAmpOutToStore = true;
    #endif
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();

    while (true) {

        while ((output_cache_read_pos + 1) % OUTPUT_BLOCKS_TO_CACHE == output_cache_write_pos) {
            std::this_thread::yield();
        }
        output_cache_read_pos = (output_cache_read_pos + 1) % OUTPUT_BLOCKS_TO_CACHE;

        uint8_t * output_block = Output + output_cache_block_size * output_cache_read_pos;
        #if SHOW_DEBUG_OUTPUT == TRUE
        uint8_t * outputFloat_block = OutputFloat + output_cache_block_size * output_cache_read_pos;
        #endif

        #if VERIFY_AMPOUT == TRUE
        sha3_ctx sha3;
        rhash_sha3_256_init(&sha3);
        rhash_sha3_update(&sha3, (const unsigned char*)&output_block[0], vertical_len / 8);
        unsigned char* hash = (unsigned char*)malloc(32);
        rhash_sha3_final(&sha3, hash);
        if (memcmp(hash, ampout_sha3, 32) == 0) {
            println("VERIFIED!")
        }
        else
        {
            println("VERIFICATION FAILED!")
            exit(101);
        }
        #endif

        #if STORE_FIRST_AMPOUT_IN_FILE == TRUE
        if (firstAmpOutToStore) {
            firstAmpOutToStore = false;
            auto ampout_file = std::fstream("ampout.bin", std::ios::out | std::ios::binary);
            ampout_file.write((char*)&output_block[0], vertical_len / 8);
            ampout_file.close();
        }
        #endif

        #if HOST_AMPOUT_SERVER == TRUE
        retry_sending_amp_out:
        rc = zmq_recv(amp_out_socket, syn, 3, 0);
        if (rc != 3 || syn[0] != 'S' || syn[1] != 'Y' || syn[2] != 'N') {
            println("Error receiving SYN! Retrying...");
            goto retry_sending_amp_out;
        }
        if (zmq_send(amp_out_socket, output_block, vertical_len / 8, 0) != vertical_len / 8) {
            println("Error sending data to AMPOUT client! Retrying...");
            goto retry_sending_amp_out;
        }
        println("Block sent to AMPOUT Client");
        #endif

        #if SHOW_DEBUG_OUTPUT == TRUE
        printlock.lock();
        for (size_t i = 0; i < min_template(dist_freq, 64); ++i)
        {
            printf("%f\n", outputFloat_block[i]);
        }
        printlock.unlock();
        #endif

        stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        start = std::chrono::high_resolution_clock::now();

        #if SHOW_AMPOUT == TRUE
        printlock.lock();
        std::cout << "Blocktime: " << duration/1000.0 << " ms => " << (1000000.0/duration)*(sample_size/1000000.0) << " Mbit/s" << std::endl;
        for (size_t i = 0; i < min_template(vertical_block * sizeof(uint32_t), 4); ++i)
        {
            printf("0x%02X: %s\n", output_block[i], std::bitset<8>(output_block[i]).to_string().c_str());
        }
        fflush(stdout);
        printlock.unlock();
        #endif
    }
}


int main(int argc, char* argv[])
{
    std::cout << "PrivacyAmplification with " << sample_size << " bits" << std::endl << std::endl;

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

    uint32_t dist_sample = sample_size;
    uint32_t dist_freq = sample_size / 2 + 1;

    input_cache_read_pos = INPUT_BLOCKS_TO_CACHE - 1;
    input_cache_write_pos = 0;
    output_cache_read_pos = INPUT_BLOCKS_TO_CACHE - 1;
    output_cache_write_pos = 0;

    uint32_t* count_one_global_seed;
    uint32_t* count_one_global_key;
    float* correction_float_dev;
    Real* di1;
    Real* di2;
    Real* invOut;
    Complex* do1;
    Complex* do2;
    cudaStream_t FFTStream, BinInt2floatKeyStream, BinInt2floatSeedStream, CalculateCorrectionFloatStream,
        cpu2gpuKeyStartStream, cpu2gpuKeyRestStream, cpu2gpuSeedStream, gpu2cpuStream, ElementWiseProductStream, ToBinaryArrayStream;
    cudaStreamCreate(&FFTStream);
    cudaStreamCreate(&BinInt2floatKeyStream);
    cudaStreamCreate(&BinInt2floatSeedStream);
    cudaStreamCreate(&CalculateCorrectionFloatStream);
    cudaStreamCreate(&cpu2gpuKeyStartStream);
    cudaStreamCreate(&cpu2gpuKeyRestStream);
    cudaStreamCreate(&cpu2gpuSeedStream);
    cudaStreamCreate(&gpu2cpuStream);
    cudaStreamCreate(&ElementWiseProductStream);
    cudaStreamCreate(&ToBinaryArrayStream);

    // create cuda event to measure the performance
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate host pinned memory on RAM
    cudaMallocHost((void**)&toeplitz_seed, input_cache_block_size * sizeof(uint32_t) * INPUT_BLOCKS_TO_CACHE);
    cudaMallocHost((void**)&key_start, input_cache_block_size * sizeof(uint32_t) * INPUT_BLOCKS_TO_CACHE);
    cudaMallocHost((void**)&key_rest, input_cache_block_size * sizeof(uint32_t) * INPUT_BLOCKS_TO_CACHE + 31 * sizeof(uint32_t));
    cudaMallocHost((void**)&Output, output_cache_block_size * OUTPUT_BLOCKS_TO_CACHE);
    #if SHOW_DEBUG_OUTPUT == TRUE
    cudaMallocHost((void**)&OutputFloat, dist_sample * sizeof(float) * OUTPUT_BLOCKS_TO_CACHE);
    #endif

    //Set key_start_zero_pos and key_rest_zero_pos to thair default values
    std::fill(key_start_zero_pos, key_start_zero_pos + INPUT_BLOCKS_TO_CACHE, desired_block);
    std::fill(key_rest_zero_pos, key_rest_zero_pos + INPUT_BLOCKS_TO_CACHE, desired_block);

    // Allocate memory on GPU
    cudaMalloc(&count_one_global_seed, sizeof(uint32_t));
    cudaMalloc(&count_one_global_key, sizeof(uint32_t));
    cudaMalloc(&correction_float_dev, sizeof(float));
    cudaCalloc((void**)&di1, sample_size * sizeof(Real));
    cudaMalloc((void**)&di2, sample_size * sizeof(Real));
    cudaMalloc((void**)&do1, sample_size * sizeof(Complex));
    cudaMalloc((void**)&do2, sample_size * sizeof(Complex));
    cudaMalloc(&invOut, (dist_sample + 992) * sizeof(Real));

    register const Complex complex0 = make_float2(0.0f, 0.0f);
    register const Real float0 = 0.0f;
    register const Real float1_reduced = 1.0f/reduction;

    cudaMemcpyToSymbol(c0_dev, &complex0, sizeof(Complex));
    cudaMemcpyToSymbol(h0_dev, &float0, sizeof(float));
    cudaMemcpyToSymbol(h1_reduced_dev, &float1_reduced, sizeof(float));
    cudaMemcpyToSymbol(normalisation_float_dev, &normalisation_float, sizeof(float));

    std::thread threadReciveObj(reciveData);
    threadReciveObj.detach();
    std::thread threadSendObj(sendData);
    threadSendObj.detach();

    cufftHandle plan_forward_R2C;
    cufftResult r;
    r = cufftPlan1d(&plan_forward_R2C, dist_sample, CUFFT_R2C, 1);
    if (r != CUFFT_SUCCESS)
    {
        printf("Failed to plan FFT 1! Error Code: %i\n", r);
        exit(0);
    }
    cufftSetStream(plan_forward_R2C, FFTStream);

    cufftHandle plan_inverse_C2R;
    r = cufftPlan1d(&plan_inverse_C2R, dist_sample, CUFFT_C2R, 1);
    if (r != CUFFT_SUCCESS)
    {
        printf("Failed to plan IFFT 1! Error Code: %i\n", r);
        exit(0);
    }
    cufftSetStream(plan_forward_R2C, FFTStream);


    uint32_t relevant_keyBlocks = horizontal_block + 1;
    uint32_t relevant_keyBlocks_old = 0;
    bool recalculate_toeplitz_matrix_seed = true;

    while (true) {

        while ((input_cache_read_pos + 1) % INPUT_BLOCKS_TO_CACHE == input_cache_write_pos) {
            std::this_thread::yield();
        }
        input_cache_read_pos = (input_cache_read_pos + 1) % INPUT_BLOCKS_TO_CACHE;

        relevant_keyBlocks_old = relevant_keyBlocks;
        relevant_keyBlocks = horizontal_block + 1;
        if (relevant_keyBlocks_old > relevant_keyBlocks) {
            cudaMemset(di1 + relevant_keyBlocks, 0b00000000, (relevant_keyBlocks_old - relevant_keyBlocks) * sizeof(Real));
        }

        cudaMemset(count_one_global_key, 0x00, sizeof(uint32_t));
        binInt2float KERNEL_ARG4((int)((relevant_keyBlocks*32+1023) / 1024), minValue(relevant_keyBlocks * 32, 1024), 0,
            BinInt2floatKeyStream) (key_start + input_cache_block_size * input_cache_read_pos, di1, count_one_global_key);
        if (recalculate_toeplitz_matrix_seed) {
            cudaMemset(count_one_global_seed, 0x00, sizeof(uint32_t));
            binInt2float KERNEL_ARG4((int)(((int)(sample_size)+1023) / 1024), std::min(sample_size, 1024), 0,
                BinInt2floatSeedStream) (toeplitz_seed + input_cache_block_size * input_cache_read_pos, di2, count_one_global_seed);
            cudaStreamSynchronize(BinInt2floatSeedStream);
        }
        cudaStreamSynchronize(BinInt2floatKeyStream);
        calculateCorrectionFloat KERNEL_ARG4(1, 1, 0, CalculateCorrectionFloatStream) (count_one_global_key, count_one_global_seed, correction_float_dev);
        cufftExecR2C(plan_forward_R2C, di1, do1);
        if (recalculate_toeplitz_matrix_seed) {
            cufftExecR2C(plan_forward_R2C, di2, do2);
        }
        cudaStreamSynchronize(FFTStream);
        cudaStreamSynchronize(CalculateCorrectionFloatStream);
        setFirstElementToZero KERNEL_ARG4(1, 2, 0, ElementWiseProductStream) (do1, do2);
        cudaStreamSynchronize(ElementWiseProductStream);
        ElementWiseProduct KERNEL_ARG4((int)((dist_freq + 1023) / 1024), std::min((int)dist_freq, 1024), 0, ElementWiseProductStream) (do1, do2);
        cudaStreamSynchronize(ElementWiseProductStream);
        cufftExecC2R(plan_inverse_C2R, do1, invOut);
        cudaStreamSynchronize(FFTStream);

        while (output_cache_write_pos % OUTPUT_BLOCKS_TO_CACHE == output_cache_read_pos) {
            std::this_thread::yield();
        }

        uint32_t* binOut = reinterpret_cast<uint32_t*>(Output + output_cache_block_size * output_cache_write_pos);
        ToBinaryArray KERNEL_ARG4((int)((int)(vertical_block) / 31) + 1, 1023, 0, ToBinaryArrayStream)
            (invOut, binOut, key_rest + input_cache_block_size * input_cache_read_pos, correction_float_dev);
        cudaStreamSynchronize(ToBinaryArrayStream);

        #if DYNAMIC_TOEPLITZ_MATRIX_SEED == FALSE
        recalculate_toeplitz_matrix_seed = false;
        #endif

        #if SHOW_DEBUG_OUTPUT == TRUE
        cudaMemcpy(OutputFloat + output_cache_block_size * output_cache_write_pos, invOut, dist_freq * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(OutputFloat + output_cache_block_size * output_cache_write_pos, correction_float_dev, sizeof(float), cudaMemcpyDeviceToHost);
        #endif

        output_cache_write_pos = (output_cache_write_pos + 1) % OUTPUT_BLOCKS_TO_CACHE;
    }


    // Delete CUFFT Plans
    cufftDestroy(plan_forward_R2C);
    cufftDestroy(plan_inverse_C2R);

    // Deallocate memoriey on GPU and RAM
    cudaFree(di1);
    cudaFree(di2);
    cudaFree(invOut);
    cudaFree(do1);
    cudaFree(do2);
    cudaFree(Output);

    // Delete cuda events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
