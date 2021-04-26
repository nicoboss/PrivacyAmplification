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
#include <random>
#include "yaml/Yaml.hpp"
#include "sha3/sha3.h"
#if !defined(__NVCC__)
#include "sha3/sha3.c"
#endif
#include "ThreadPool.h"



#define VKFFT_BACKEND 0
//#define __NVCC__

#if defined(__NVCC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#else
#include "vulkan/vulkan.h"
#include "glslang_c_interface.h"
//#include "half_lib/half.hpp"
#include <vuda/vuda_runtime.hpp>
#endif
#include "vkFFT/vkFFT.h"
#include "vkFFT/vkFFT_helper.h"

#if !defined(NDEBUG)
#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED
#endif

#include "PrivacyAmplification.h"
//#define __NVCC__

using namespace std;


//Little endian only!
//#define TEST

#ifdef __CUDACC__
#define KERNEL_ARG2(grid, block) <<< grid, block >>>
#define KERNEL_ARG3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARG2(grid, block)
#define KERNEL_ARG3(grid, block, sh_mem)
#define KERNEL_ARG4(grid, block, sh_mem, stream)
#endif

#if defined(__NVCC__)
#ifdef __INTELLISENSE__
cudaError_t cudaMemcpyToSymbol(Complex symbol, const void* src, size_t count);
cudaError_t cudaMemcpyToSymbol(Real symbol, const void* src, size_t count);
int __float2int_rn(float in);
unsigned int atomicAdd(unsigned int* address, unsigned int val);
#define __syncthreads()
#endif
#endif

#ifdef DEBUG
	#ifdef _WIN32
		#define BREAK __debugbreak();
	#else
		#define BREAK __builtin_trap();
	#endif
#else
	#define BREAK
#endif

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define assertZeroThreshold(actual, threshold, testCaseNr) \
if (abs(actual) > threshold) { \
	std::cerr << "AssertionError in function " << __func__ << " in " << __FILENAME__ << ":" << __LINE__ << " on test case " << testCaseNr \
			  << ": Expected abs(" << actual << ") < " << threshold << endl; \
	unitTestsFailed = true;  \
	unitTestsFailedLocal = true; \
	BREAK \
}

#define assertEquals(actual, expected, testCaseNr) \
if (actual != expected) { \
	std::cerr << "AssertEqualsError in function " << __func__ << " in " << __FILENAME__ << ":" << __LINE__ << " on test case " << testCaseNr \
			  << ": Expected " << expected << " but it was " << actual << endl; \
	unitTestsFailed = true;  \
	unitTestsFailedLocal = true; \
	BREAK \
}

#define assertTrue(actual) \
if (!(actual)) { \
	std::cerr << "AssertTrueError in function " << __func__ << " in " << __FILE__ << ":" << __LINE__ << endl; \
	BREAK \
	exit(101); \
}

#if defined(__NVCC__)
#define assertGPU(data, data_len, value) \
cudaDeviceSynchronize(); \
cudaAssertValue KERNEL_ARG3(max(data_len / 1024, 1), min(data_len, 1024), 0) (data, value); \
{ \
	cudaError_t error = cudaDeviceSynchronize(); \
	assertTrue(error == cudaSuccess); \
}
#else
#define assertGPU(data, data_len, value) \
*assertKernelValue = value; \
*assertKernelReturnValue = 0; \
vuda::launchKernel("SPIRV/assert.spv", "main", 0, max(data_len / 1024, 1), min(data_len, 1024), data, assertKernelValue, assertKernelReturnValue); \
cudaStreamSynchronize(0); \
assertTrue(*assertKernelReturnValue == 0);
#endif

#if defined(__NVCC__)
/*Because cudaCalloc doesn't exist let's make our own one using cudaMalloc and cudaMemset*/
#define cudaCalloc(address, size) if (cudaMalloc(address, size) == cudaSuccess) cudaMemset(*address, 0b00000000, size);
#else
#define cudaMemset(address, value, num) \
*value_dev = value; \
vuda::launchKernel("SPIRV/memset.spv", "main", 0, max(num / 1024, 1), min(num, 1024), value_dev, address); \
cudaStreamSynchronize(0);
#define cudaCalloc(address, size) \
if (cudaMalloc(address, size) == cudaSuccess) cudaMemset(*address, 0b00000000, size);
#endif

#define VULKAN_ASSERT_VALUE(data, data_len, value) \
cudaDeviceSynchronize(); \
cudaAssertValue KERNEL_ARG3(max(data_len/1024, 1), data_len, 0) (data, value); \
{ \
cudaError_t error = cudaDeviceSynchronize(); \
assertTrue(error == cudaSuccess); \
}

#if STOPWATCH == TRUE
chrono::steady_clock::time_point start;
chrono::steady_clock::time_point checkpoint;
chrono::steady_clock::time_point new_checkpoint;

#define STOPWATCH_START \
start = std::chrono::high_resolution_clock::now(); \
checkpoint = start;

#define STOPWATCH_SAVE(VALUE) \
new_checkpoint = std::chrono::high_resolution_clock::now(); \
VALUE = std::chrono::duration_cast<std::chrono::nanoseconds>(new_checkpoint-checkpoint).count(); \
checkpoint = new_checkpoint;

#define STOPWATCH_TOTAL(VALUE) \
VALUE = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-start).count();
#else
#define STOPWATCH_START
#define STOPWATCH_SAVE(VALUE)
#define STOPWATCH_TOTAL(VALUE)
#endif

string address_seed_in;
string address_key_in;
string address_amp_out;

uint32_t vertical_len;
uint32_t horizontal_len;
uint32_t vertical_block;
uint32_t horizontal_block;
uint32_t desired_block;
uint32_t key_blocks;
uint32_t input_cache_block_size;
uint32_t output_cache_block_size;
uint32_t* recv_key;
uint32_t* toeplitz_seed;
uint32_t* key_start;
uint32_t* key_start_zero_pos;
uint32_t* key_rest;
uint32_t* key_rest_zero_pos;
uint8_t* Output;
uint32_t* assertKernelValue;
uint32_t* assertKernelReturnValue;
uint8_t* testMemoryHost;

#if SHOW_DEBUG_OUTPUT == TRUE
Real* OutputFloat;
#endif
atomic<uint32_t> input_cache_read_pos;
atomic<uint32_t> input_cache_write_pos;
atomic<uint32_t> output_cache_read_pos;
atomic<uint32_t> output_cache_write_pos;
mutex printlock;
float normalisation_float;
atomic<bool> unitTestsFailed = false;
atomic<bool> unitTestBinInt2floatVerifyResultThreadFailed = false;
atomic<bool> unitTestToBinaryArrayVerifyResultThreadFailed = false;
atomic<bool> cuFFT_planned = false;

#if defined(__NVCC__)
__device__ __constant__ Complex c0_dev;
__device__ __constant__ Real h0_dev;
__device__ __constant__ Real h1_reduced_dev;
__device__ __constant__ Real normalisation_float_dev;
__device__ __constant__ uint32_t sample_size_dev;
__device__ __constant__ uint32_t pre_mul_reduction_dev;

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
#endif


void printStream(ostream& os) {
	ostringstream& ss = dynamic_cast<ostringstream&>(os);
	printlock.lock();
	cout << ss.str() << flush;
	printlock.unlock();
}


void printlnStream(ostream& os) {
	ostringstream& ss = dynamic_cast<ostringstream&>(os);
	printlock.lock();
	cout << ss.str() << endl;
	printlock.unlock();
}


string convertStreamToString(ostream& os) {
	ostringstream& ss = dynamic_cast<ostringstream&>(os);
	return ss.str();
}

#if defined(__NVCC__)
__global__ void cudaAssertValue(uint32_t* data, uint32_t value) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(data[i] == value);
}
#endif

int unitTestCalculateCorrectionFloat() {
	println("Started CalculateCorrectionFloat Unit Test...");
	bool unitTestsFailedLocal = false;
	#if defined(__NVCC__)
	cudaStream_t CalculateCorrectionFloatTestStream;
	cudaStreamCreate(&CalculateCorrectionFloatTestStream);
	#else
	const int CalculateCorrectionFloatTestStream = 0;
	#endif
	uint32_t* count_one_of_global_seed_test;
	uint32_t* count_one_of_global_key_test;
	float* correction_float_dev_test;
	uint32_t* sample_size_test;
	cudaMallocHost((void**)&count_one_of_global_seed_test, sizeof(uint32_t));
	cudaMallocHost((void**)&count_one_of_global_key_test, sizeof(uint32_t));
	cudaMallocHost((void**)&correction_float_dev_test, sizeof(float));
	cudaMallocHost((void**)&sample_size_test, sizeof(uint32_t));
	*sample_size_test = pow(2, 6);
	#if defined(__NVCC__)
	cudaMemcpyToSymbol(sample_size_dev, sample_size_test, sizeof(uint32_t));
	#endif
	for (uint32_t i = 0; i < *sample_size_test; ++i) {
		for (uint32_t j = 0; j < *sample_size_test; ++j) {
			*count_one_of_global_seed_test = i;
			*count_one_of_global_key_test = j;
			#if defined(__NVCC__)
			calculateCorrectionFloat KERNEL_ARG4(1, 1, 0, CalculateCorrectionFloatTestStream)(count_one_of_global_seed_test, count_one_of_global_key_test, correction_float_dev_test);
			#else
			vuda::launchKernel("SPIRV/calculateCorrectionFloat.spv", "main", CalculateCorrectionFloatTestStream, 1, 1, count_one_of_global_seed_test, count_one_of_global_key_test, correction_float_dev_test, sample_size_test);
			#endif
			cudaStreamSynchronize(CalculateCorrectionFloatTestStream);
			uint64_t cpu_count_multiplied = *count_one_of_global_seed_test * *count_one_of_global_key_test;
			double cpu_count_multiplied_normalized = cpu_count_multiplied / (double)*sample_size_test;
			double count_multiplied_normalized_modulo = fmod(cpu_count_multiplied_normalized, 2.0);
			assertZeroThreshold(*correction_float_dev_test - count_multiplied_normalized_modulo, 0.0001, i * *sample_size_test + j);
		}
	}
	*sample_size_test = pow(2, 27);
	#if defined(__NVCC__)
	cudaMemcpyToSymbol(sample_size_dev, sample_size_test, sizeof(uint32_t));
	#endif
	std::mt19937_64 gen(777);
	std::uniform_int_distribution<uint32_t> distrib(pow(2, 25), pow(2, 27));
	for (uint32_t n = 0; n < 4096; ++n) {
		*count_one_of_global_seed_test = distrib(gen);
		*count_one_of_global_key_test = distrib(gen);
		#if defined(__NVCC__)
		calculateCorrectionFloat KERNEL_ARG4(1, 1, 0, CalculateCorrectionFloatTestStream)(count_one_of_global_seed_test, count_one_of_global_key_test, correction_float_dev_test);
		#else
		vuda::launchKernel("SPIRV/calculateCorrectionFloat.spv", "main", CalculateCorrectionFloatTestStream, 1, 1, count_one_of_global_seed_test, count_one_of_global_key_test, correction_float_dev_test, sample_size_test);
		#endif
		cudaStreamSynchronize(CalculateCorrectionFloatTestStream);
		uint64_t cpu_count_multiplied = *count_one_of_global_seed_test * *count_one_of_global_key_test;
		double cpu_count_multiplied_normalized = cpu_count_multiplied / (double)*sample_size_test;
		double count_multiplied_normalized_modulo = fmod(cpu_count_multiplied_normalized, 2.0);
		assertZeroThreshold(*correction_float_dev_test - count_multiplied_normalized_modulo, 0.0001, n);
	}
	#if defined(__NVCC__)
	cudaMemcpyToSymbol(sample_size_dev, &sample_size, sizeof(uint32_t));
	#endif
	println("Completed CalculateCorrectionFloat Unit Test");
	return unitTestsFailedLocal ? 100 : 0;
}

#if defined(__NVCC__)
__global__
void calculateCorrectionFloat(uint32_t* count_one_of_global_seed, uint32_t* count_one_of_global_key, float* correction_float_dev)
{
	uint64_t count_multiplied = *count_one_of_global_seed * *count_one_of_global_key;
	double count_multiplied_normalized = count_multiplied / (double)sample_size_dev;
	double two = 2.0;
	Real count_multiplied_normalized_modulo = (float)fmod(count_multiplied_normalized, two);
	*correction_float_dev = count_multiplied_normalized_modulo;
}
#endif


int unitTestSetFirstElementToZero() {
	println("Started SetFirstElementToZero Unit Test...");
	bool unitTestsFailedLocal = false;
	#if defined(__NVCC__)
	cudaStream_t SetFirstElementToZeroStreamTest;
	cudaStreamCreate(&SetFirstElementToZeroStreamTest);
	#else
	const int SetFirstElementToZeroStreamTest = 0;
	#endif
	float* do1_test;
	float* do2_test;
	cudaMallocHost((void**)&do1_test, pow(2, 10) * 2 * sizeof(float));
	cudaMallocHost((void**)&do2_test, pow(2, 10) * 2 * sizeof(float));
	for (int i = 0; i < pow(2, 10) * 2; ++i) {
		do1_test[i] = i + 0.77;
		do2_test[i] = i + 0.88;
	}
	#if defined(__NVCC__)
	setFirstElementToZero KERNEL_ARG4(1, 2, 0, SetFirstElementToZeroStreamTest)(reinterpret_cast<Complex*>(do1_test), reinterpret_cast<Complex*>(do2_test));
	#else
	vuda::launchKernel("SPIRV/setFirstElementToZero.spv", "main", SetFirstElementToZeroStreamTest, 1, 2, do1_test, do2_test);
	#endif
	cudaStreamSynchronize(SetFirstElementToZeroStreamTest);
	assertZeroThreshold(do1_test[0], 0.00001, 0);
	assertZeroThreshold(do1_test[1], 0.00001, 1);
	assertZeroThreshold(do2_test[0], 0.00001, 2);
	assertZeroThreshold(do2_test[1], 0.00001, 3);
	for (int i = 2; i < pow(2, 10) * 2; ++i) {
		assertZeroThreshold(do1_test[i] - (i + 0.77), 0.0001, i * 2);
		assertZeroThreshold(do2_test[i] - (i + 0.88), 0.0001, i * 2 + 1);
	}
	println("Completed SetFirstElementToZero Unit Test");
	return unitTestsFailedLocal ? 100 : 0;
}

#if defined(__NVCC__)
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
#endif


int unitTestElementWiseProduct() {
	println("Started ElementWiseProduct Unit Test...");
	bool unitTestsFailedLocal = false;
	#if defined(__NVCC__)
	cudaStream_t ElementWiseProductStreamTest;
	cudaStreamCreate(&ElementWiseProductStreamTest);
	#else
	const int ElementWiseProductStreamTest = 0;
	#endif
	uint32_t r = pow(2, 5);
	float* do1_test;
	float* do2_test;
	uint32_t* pre_mul_reduction_test;
	cudaMallocHost((void**)&pre_mul_reduction_test, sizeof(uint32_t));
	*pre_mul_reduction_test = r;
	#if defined(__NVCC__)
	cudaMemcpyToSymbol(pre_mul_reduction_dev, pre_mul_reduction_test, sizeof(uint32_t));
	#endif
	cudaMallocHost((void**)&do1_test, pow(2, 10) * 2 * sizeof(float));
	cudaMallocHost((void**)&do2_test, pow(2, 10) * 2 * sizeof(float));
	for (int i = 0; i < pow(2, 10) * 2; ++i) {
		do1_test[i] = i + 0.77;
		do2_test[i] = i + 0.88;
	}
	#if defined(__NVCC__)
	ElementWiseProduct KERNEL_ARG4((int)((pow(2, 10) + 1023) / 1024), min((int)pow(2, 10), 1024), 0, ElementWiseProductStreamTest)(reinterpret_cast<Complex*>(do1_test), reinterpret_cast<Complex*>(do2_test));
	#else
	vuda::launchKernel("SPIRV/elementWiseProduct.spv", "main", ElementWiseProductStreamTest, (int)((pow(2, 10) + 1023) / 1024), min((int)pow(2, 10), 1024), do1_test, do2_test, pre_mul_reduction_test);
	#endif
	cudaStreamSynchronize(ElementWiseProductStreamTest);
	for (int i = 0; i < pow(2, 10) * 2; i += 2) {
		float real = ((i + 0.77) / r) * ((i + 0.88) / r) - (((i + 1) + 0.77) / r) * (((i + 1) + 0.88) / r);
		float imag = ((i + 0.77) / r) * (((i + 1) + 0.88) / r) + (((i + 1) + 0.77) / r) * ((i + 0.88) / r);
		assertZeroThreshold(do1_test[i] - real, 0.001, i);
		assertZeroThreshold(do1_test[i + 1] - imag, 0.001, i + 1);
	}
	#if defined(__NVCC__)
	cudaMemcpyToSymbol(pre_mul_reduction_dev, &pre_mul_reduction, sizeof(uint32_t));
	#endif
	println("Completed ElementWiseProduct Unit Test");
	return unitTestsFailedLocal ? 100 : 0;
}

#if defined(__NVCC__)
__global__
void ElementWiseProduct(Complex* do1, Complex* do2)
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	float r = pre_mul_reduction_dev;
	Real do1x = do1[i].x / r;
	Real do1y = do1[i].y / r;
	Real do2x = do2[i].x / r;
	Real do2y = do2[i].y / r;
	do1[i].x = do1x * do2x - do1y * do2y;
	do1[i].y = do1x * do2y + do1y * do2x;
}
#endif


//David W. Wilson: https://oeis.org/A000788/a000788.txt
unsigned A000788(unsigned n)
{
	unsigned v = 0;
	for (unsigned bit = 1; bit <= n; bit <<= 1)
		v += ((n >> 1) & ~(bit - 1)) + ((n & bit) ? (n & ((bit << 1) - 1)) - (bit - 1) : 0);
	return v;
}

void unitTestBinInt2floatVerifyResultThread(float* floatOutTest, int i, int i_max)
{
	bool unitTestsFailedLocal = false;
	const Real float0 = 0.0f;
	const Real float1_reduced = 1.0f / reduction;
	for (; i < i_max; ++i) {
		if (((i / 32) & (1 << (31 - (i % 32)))) == 0) {
			assertEquals(floatOutTest[i], float0, i)
		}
		else
		{
			assertEquals(floatOutTest[i], float1_reduced, i)
		}
	}
	if (unitTestsFailedLocal) {
		unitTestBinInt2floatVerifyResultThreadFailed = true;
	}
}

int unitTestBinInt2float() {
	println("Started TestBinInt2float Unit Test...");
	atomic<bool> unitTestsFailedLocal = false;
	#if defined(__NVCC__)
	cudaStream_t BinInt2floatStreamTest;
	cudaStreamCreate(&BinInt2floatStreamTest);
	#else
	const int BinInt2floatStreamTest = 0;
	float* float1_reduced_test_dev;
	cudaMallocHost((void**)&float1_reduced_test_dev, sizeof(float));
	float float1_reduced_test = 1.0f / reduction;
	*float1_reduced_test_dev = float1_reduced_test;
	#endif
	uint32_t* binInTest;
	float* floatOutTest;
	cudaMallocHost((void**)&binInTest, (pow(2, 27) / 32) * sizeof(uint32_t));
	cudaMallocHost((void**)&floatOutTest, pow(2, 27) * sizeof(float));
	uint32_t* count_one_test;
	cudaMallocHost((void**)&count_one_test, sizeof(uint32_t));

	const auto processor_count = std::thread::hardware_concurrency();
	for (int i = 0; i < pow(2, 27) / 32; ++i) {
		binInTest[i] = i;
	}
	unitTestBinInt2floatVerifyResultThreadFailed = false;
	for (uint32_t sample_size_test_exponent = 10; sample_size_test_exponent <= 27; ++sample_size_test_exponent)
	{
		int elementsToCheck = pow(2, sample_size_test_exponent);
		println("TestBinInt2float Unit Test with 2^" << sample_size_test_exponent << " samples...");
		uint32_t sample_size_test = elementsToCheck;
		uint32_t count_one_expected = A000788((sample_size_test / 32) - 1);
		*count_one_test = 0;
		memset(floatOutTest, 0xFF, pow(2, 27) * sizeof(float));
		#if defined(__NVCC__)
		binInt2float KERNEL_ARG4((int)(((int)(sample_size_test)+1023) / 1024), min_template(sample_size_test, 1024), 0, BinInt2floatStreamTest) (binInTest, floatOutTest, count_one_test);
		#else
		vuda::launchKernel("SPIRV/binInt2float.spv", "main", BinInt2floatStreamTest, (int)(((int)(sample_size_test)+1023) / 1024), min_template(sample_size_test, 1024), binInTest, floatOutTest, count_one_test, float1_reduced_test_dev);
	#endif
		cudaStreamSynchronize(BinInt2floatStreamTest);
		assertEquals(*count_one_test, count_one_expected, -1);
		int requiredTotalTasks = elementsToCheck % 1000000 == 0 ? elementsToCheck / 1000000 : (elementsToCheck / 1000000) + 1;
		ThreadPool* unitTestBinInt2floatVerifyResultPool = new ThreadPool(min(max(processor_count, 1), requiredTotalTasks));
		for (int i = 0; i < elementsToCheck; i += 1000000) {
			unitTestBinInt2floatVerifyResultPool->enqueue(unitTestBinInt2floatVerifyResultThread, floatOutTest, i, min(i + 1000000, elementsToCheck));
		}
		unitTestBinInt2floatVerifyResultPool->~ThreadPool();
	}
	if (unitTestBinInt2floatVerifyResultThreadFailed) {
		unitTestsFailedLocal = true;
	}
	println("Completed TestBinInt2float Unit Test");
	return unitTestsFailedLocal ? 100 : 0;
}

#if defined(__NVCC__)
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
#endif


void unitTestToBinaryArrayVerifyResultThread(uint32_t* binOutTest, uint32_t* key_rest_test, int i, int i_max)
{
	bool unitTestsFailedLocal = false;
	uint32_t mask;
	uint32_t data;
	uint32_t key_rest_little;
	uint32_t key_rest_xor;
	uint32_t actualBit;
	uint32_t expectedBit;
	uint32_t xorBit;
	for (; i < i_max; ++i) {
		mask = 1 << (31 - (i % 32));
		data = binOutTest[i / 32];
		#if AMPOUT_REVERSE_ENDIAN == TRUE
		data = ((((data) & 0xff000000) >> 24) |
			(((data) & 0x00ff0000) >> 8) |
			(((data) & 0x0000ff00) << 8) |
			(((data) & 0x000000ff) << 24));
		#endif
		#if XOR_WITH_KEY_REST == TRUE
		#if AMPOUT_REVERSE_ENDIAN == TRUE
		key_rest_little = key_rest_test[i / 32];
		key_rest_xor = ((((key_rest_little) & 0xff000000) >> 24) |
			(((key_rest_little) & 0x00ff0000) >> 8) |
			(((key_rest_little) & 0x0000ff00) << 8) |
			(((key_rest_little) & 0x000000ff) << 24));
		#else
		uint32_t key_rest_xor = key_rest_test[i / 32];
		#endif
		#endif
		actualBit = (data & mask) > 0;
		expectedBit = ((i / 32) & mask) > 0;
		xorBit = (key_rest_xor & mask) > 0;
		#if XOR_WITH_KEY_REST
		expectedBit ^= xorBit;
		#endif
		assertEquals(actualBit, expectedBit, i)
	}
	if (unitTestsFailedLocal) {
		unitTestToBinaryArrayVerifyResultThreadFailed = true;
	}
}

int unitTestToBinaryArray() {
	println("Started ToBinaryArray Unit Test...");
	atomic<bool> unitTestsFailedLocal = false;
	#if defined(__NVCC__)
	cudaStream_t ToBinaryArrayStreamTest;
	cudaStreamCreate(&ToBinaryArrayStreamTest);
	#else
	const int ToBinaryArrayStreamTest = 0;
	#endif
	const Real float0 = 0.0f;
	const Real float1 = 1.0f;
	float* invOutTest;
	uint32_t* binOutTest;
	uint32_t* key_rest_test;
	Real* correction_float_dev_test;
	cudaMallocHost((void**)&invOutTest, pow(2, 27) * sizeof(float));
	cudaMallocHost((void**)&binOutTest, (pow(2, 27) / 32) * sizeof(uint32_t));
	cudaMallocHost((void**)&key_rest_test, (pow(2, 27) / 32) * sizeof(uint32_t));
	cudaMallocHost((void**)&correction_float_dev_test, sizeof(Real));
	memset(key_rest_test, 0b10101010, (pow(2, 27) / 32) * sizeof(uint32_t));
	*correction_float_dev_test = 1.9f;
	#if defined(__NVCC__)
	float normalisation_float_test = 1.0f;
	cudaMemcpyToSymbol(normalisation_float_dev, &normalisation_float_test, sizeof(float));
	#else
	float* normalisation_float_test_dev;
	cudaMallocHost((void**)&normalisation_float_test_dev, sizeof(float));
	*normalisation_float_test_dev = 1.0f;
	#endif
	const auto processor_count = std::thread::hardware_concurrency();
	for (int i = 0; i < pow(2, 27); ++i) {
		invOutTest[i] = (((i / 32) & (1 << (31 - (i % 32)))) == 0) ? float0 : float1;
	}
	unitTestToBinaryArrayVerifyResultThreadFailed = false;
	for (uint32_t sample_size_test_exponent = 10; sample_size_test_exponent <= 27; ++sample_size_test_exponent)
	{
		uint32_t sample_size_test = pow(2, sample_size_test_exponent);
		uint32_t vertical_len_test = sample_size_test / 4 + sample_size_test / 8;
		uint32_t elementsToCheck = vertical_len_test;
		uint32_t vertical_block_test = vertical_len_test / 32;
		println("ToBinaryArray Unit Test with 2^" << sample_size_test_exponent << " samples...");
		memset(binOutTest, 0xCC, (pow(2, 27) / 32) * sizeof(uint32_t));
		#if defined(__NVCC__)
		ToBinaryArray KERNEL_ARG4((int)((int)(vertical_block_test) / 31) + 1, 1023, 0, ToBinaryArrayStreamTest) (invOutTest, binOutTest, key_rest_test, correction_float_dev_test);
		#else
		vuda::launchKernel("SPIRV/toBinaryArray.spv", "main", ToBinaryArrayStreamTest, (int)((int)(vertical_block_test) / 31) + 1, 1023, invOutTest, binOutTest, key_rest_test, correction_float_dev_test, normalisation_float_test_dev);
		#endif
		cudaStreamSynchronize(ToBinaryArrayStreamTest);
		int requiredTotalTasks = elementsToCheck % 1000000 == 0 ? elementsToCheck / 1000000 : (elementsToCheck / 1000000) + 1;
		ThreadPool* unitTestToBinaryArrayVerifyResultPool = new ThreadPool(min(max(processor_count, 1), requiredTotalTasks));
		for (int i = 0; i < elementsToCheck; i += 1000000) {
			unitTestToBinaryArrayVerifyResultPool->enqueue(unitTestToBinaryArrayVerifyResultThread, binOutTest, key_rest_test, i, min(i + 1000000, elementsToCheck));
		}
		unitTestToBinaryArrayVerifyResultPool->~ThreadPool();
	}
	if (unitTestToBinaryArrayVerifyResultThreadFailed) {
		unitTestsFailedLocal = true;
	}
	#if defined(__NVCC__)
	cudaMemcpyToSymbol(normalisation_float_dev, &normalisation_float, sizeof(uint32_t));
	#endif
	println("Completed ToBinaryArray Unit Test");
	return unitTestsFailedLocal ? 100 : 0;
}

#if defined(__NVCC__)
__global__
void ToBinaryArray(Real* invOut, uint32_t* binOut, uint32_t* key_rest_local, Real* correction_float_dev)
{
	const Real normalisation_float_local = normalisation_float_dev;
	const uint32_t block = blockIdx.x;
	const uint32_t idx = threadIdx.x;
	const Real correction_float = *correction_float_dev;

	#if XOR_WITH_KEY_REST == TRUE
	__shared__ uint32_t key_rest_xor[31];
	#endif
	__shared__ uint32_t binOutRawBit[992];
	if (idx < 992) {
		binOutRawBit[idx] = ((__float2int_rn(invOut[block * 992 + idx] / normalisation_float_local + correction_float) & 1)
			<< ToBinaryBitShiftArray_dev[idx % 32]);
	}
	else if (idx < 1023)
	{
		#if XOR_WITH_KEY_REST == TRUE
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
#endif


void printBin(const uint8_t* position, const uint8_t* end) {
	while (position < end) {
		printf("%s", bitset<8>(*position).to_string().c_str());
		++position;
	}
	cout << endl;
}


void printBin(const uint32_t* position, const uint32_t* end) {
	while (position < end) {
		printf("%s", bitset<32>(*position).to_string().c_str());
		++position;
	}
	cout << endl;
}

void memdump(string const& filename, void* data, size_t const bytes)
{
	fstream myfile = fstream(filename.c_str(), std::ios::out | std::ios::binary);
	myfile.write(reinterpret_cast<char const*>(data), bytes);
	myfile.close();
}

pair<double, double> FletcherFloat(float* data, int count)
{
	double sum1 = 0.0;
	double sum2 = 0.0;
	 
	for (int index = 0; index < count; ++index)
	{
		sum1 += (double)abs(data[index]);
		sum2 += sum1;
	}

	return make_pair(sum1, sum2);
}

bool isFletcherFloat(float* data, int count, const double expectedSum1, const double allowedAbsDeltaSum1, const double expectedSum2, const double allowedAbsDeltaSum2) {
	pair<double, double> result = FletcherFloat(data, count);
	println(std::fixed << std::setprecision(8) << result.first << " | " << result.second);
	return abs(result.first - expectedSum1) < allowedAbsDeltaSum1 && abs(result.second - expectedSum2) < allowedAbsDeltaSum2;
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


inline void readMatrixSeedFromFile() {
	//Cryptographically random Toeplitz seed generated by XOR a self-generated
	//VeraCrypt key file (PRF: SHA-512) with ANU_20Oct2017_100MB_7
	//from the ANU Quantum Random Numbers Server (https://qrng.anu.edu.au/)
	ifstream seedfile(toeplitz_seed_path, ios::binary);

	if (seedfile.fail())
	{
		cout << "Can't open file \"" << toeplitz_seed_path << "\" => terminating!" << endl;
		exit(103);
		abort();
	}

	seedfile.seekg(0, ios::end);
	size_t seedfile_length = seedfile.tellg();
	seedfile.seekg(0, ios::beg);

	if (seedfile_length < desired_block * sizeof(uint32_t))
	{
		cout << "File \"" << toeplitz_seed_path << "\" is with " << seedfile_length << " bytes too short!" << endl;
		cout << "it is required to be at least " << desired_block * sizeof(uint32_t) << " bytes => terminating!" << endl;
		exit(104);
		abort();
	}

	char* toeplitz_seed_char = reinterpret_cast<char*>(toeplitz_seed + input_cache_block_size * input_cache_write_pos);
	seedfile.read(toeplitz_seed_char, desired_block * sizeof(uint32_t));
	for (uint32_t i = 0; i < input_blocks_to_cache; ++i) {
		uint32_t* toeplitz_seed_block = toeplitz_seed + input_cache_block_size * i;
		memcpy(toeplitz_seed_block, toeplitz_seed, input_cache_block_size * sizeof(uint32_t));
	}
}


inline void readKeyFromFile() {
	//Cryptographically random Toeplitz seed generated by XOR a self-generated
	//VeraCrypt key file (PRF: SHA-512) with ANU_20Oct2017_100MB_49
	//from the ANU Quantum Random Numbers Server (https://qrng.anu.edu.au/)
	ifstream keyfile(keyfile_path, ios::binary);

	if (keyfile.fail())
	{
		cout << "Can't open file \"" << keyfile_path << "\" => terminating!" << endl;
		exit(105);
		abort();
	}

	keyfile.seekg(0, ios::end);
	size_t keyfile_length = keyfile.tellg();
	keyfile.seekg(0, ios::beg);

	if (keyfile_length < key_blocks * sizeof(uint32_t))
	{
		cout << "File \"" << keyfile_path << "\" is with " << keyfile_length << " bytes too short!" << endl;
		cout << "it is required to be at least " << key_blocks * sizeof(uint32_t) << " bytes => terminating!" << endl;
		exit(106);
		abort();
	}

	char* recv_key_char = reinterpret_cast<char*>(recv_key);
	keyfile.read(recv_key_char, key_blocks * sizeof(uint32_t));
	key2StartRest();
	for (uint32_t i = 0; i < input_blocks_to_cache; ++i) {
		uint32_t* key_start_block = key_start + input_cache_block_size * i;
		uint32_t* key_rest_block = key_rest + input_cache_block_size * i;
		uint32_t* key_start_zero_pos_block = key_start_zero_pos + i;
		uint32_t* key_rest_zero_pos_block = key_rest_zero_pos + i;
		memcpy(key_start_block, key_start, input_cache_block_size * sizeof(uint32_t));
		memcpy(key_rest_block, key_rest, input_cache_block_size * sizeof(uint32_t));
		*key_start_zero_pos_block = *key_start_zero_pos;
		*key_rest_zero_pos_block = *key_rest_zero_pos;
	}
}


void reciveData() {
	void* socket_seed_in = nullptr;
	void* socket_key_in = nullptr;
	void* context_seed_in = nullptr;
	void* context_key_in = nullptr;
	int timeout_seed_in = 1000;
	int timeout_key_in = 1000;

	if (use_matrix_seed_server)
	{
		context_seed_in = zmq_ctx_new();
		socket_seed_in = zmq_socket(context_seed_in, ZMQ_REQ);
		zmq_setsockopt(socket_seed_in, ZMQ_RCVTIMEO, &timeout_seed_in, sizeof(int));
		zmq_connect(socket_seed_in, address_seed_in.c_str());
	}
	else
	{
		readMatrixSeedFromFile();
	}

	if (use_key_server)
	{
		context_key_in = zmq_ctx_new();
		socket_key_in = zmq_socket(context_key_in, ZMQ_REQ);
		zmq_setsockopt(socket_key_in, ZMQ_RCVTIMEO, &timeout_key_in, sizeof(int));
		zmq_connect(socket_key_in, address_key_in.c_str());
	}
	else
	{
		readKeyFromFile();
	}

	bool recive_toeplitz_matrix_seed = use_matrix_seed_server;
	while (true)
	{

		while (input_cache_write_pos % input_blocks_to_cache == input_cache_read_pos) {
			this_thread::yield();
		}

		uint32_t* toeplitz_seed_block = toeplitz_seed + input_cache_block_size * input_cache_write_pos;
		if (recive_toeplitz_matrix_seed) {
		retry_receiving_seed:
			zmq_send(socket_seed_in, "SYN", 3, 0);
			if (zmq_recv(socket_seed_in, toeplitz_seed_block, desired_block * sizeof(uint32_t), 0) != desired_block * sizeof(uint32_t)) {
				println("Error receiving data from Seedserver! Retrying...");
				zmq_close(context_seed_in);
				socket_seed_in = zmq_socket(context_seed_in, ZMQ_REQ);
				zmq_setsockopt(socket_seed_in, ZMQ_RCVTIMEO, &timeout_seed_in, sizeof(int));
				zmq_connect(socket_seed_in, address_seed_in.c_str());
				goto retry_receiving_seed;
			}
			if (show_zeromq_status) {
				println("Seed Block recived");
			}

			if (!dynamic_toeplitz_matrix_seed)
			{
				recive_toeplitz_matrix_seed = false;
				zmq_disconnect(socket_seed_in, address_seed_in.c_str());
				zmq_close(socket_seed_in);
				zmq_ctx_destroy(socket_seed_in);
				for (uint32_t i = 0; i < input_blocks_to_cache; ++i) {
					uint32_t* toeplitz_seed_block = toeplitz_seed + input_cache_block_size * i;
					memcpy(toeplitz_seed_block, toeplitz_seed, input_cache_block_size * sizeof(uint32_t));
				}
			}
		}

		if (use_key_server)
		{
		retry_receiving_key:
			if (zmq_send(socket_key_in, "SYN", 3, 0) != 3) {
				println("Error sending SYN to Keyserver! Retrying...");
				goto retry_receiving_key;
			}
			if (zmq_recv(socket_key_in, &vertical_block, sizeof(uint32_t), 0) != sizeof(uint32_t)) {
				println("Error receiving vertical_blocks from Keyserver! Retrying...");
				zmq_close(context_key_in);
				socket_key_in = zmq_socket(context_key_in, ZMQ_REQ);
				zmq_setsockopt(socket_key_in, ZMQ_RCVTIMEO, &timeout_key_in, sizeof(int));
				zmq_connect(socket_key_in, address_key_in.c_str());
				goto retry_receiving_key;
			}
			vertical_len = vertical_block * 32;
			horizontal_len = sample_size - vertical_len;
			horizontal_block = horizontal_len / 32;
			if (zmq_recv(socket_key_in, recv_key, key_blocks * sizeof(uint32_t), 0) != key_blocks * sizeof(uint32_t)) {
				println("Error receiving data from Keyserver! Retrying...");
				zmq_close(context_key_in);
				socket_key_in = zmq_socket(context_key_in, ZMQ_REQ);
				zmq_setsockopt(socket_key_in, ZMQ_RCVTIMEO, &timeout_key_in, sizeof(int));
				zmq_connect(socket_key_in, address_key_in.c_str());
				goto retry_receiving_key;
			}
			if (show_zeromq_status) {
				println("Key Block recived");
			}
			key2StartRest();
		}

		#if SHOW_INPUT_DEBUG_OUTPUT == TRUE
		uint32_t* key_start_block = key_start + input_cache_block_size * input_cache_write_pos;
		uint32_t* key_rest_block = key_rest + input_cache_block_size * input_cache_write_pos;
		printlock.lock();
		cout << "Toeplitz Seed: ";
		printBin(toeplitz_seed_block, toeplitz_seed_block + desired_block);
		cout << "Key: ";
		printBin(recv_key, recv_key + key_blocks);
		cout << "Key Start: ";
		printBin(key_start_block, key_start_block + desired_block + 1);
		cout << "Key Rest: ";
		printBin(key_rest_block, key_rest_block + vertical_block + 1);
		fflush(stdout);
		printlock.unlock();
		#endif

		input_cache_write_pos = (input_cache_write_pos + 1) % input_blocks_to_cache;
	}

	if (use_matrix_seed_server && recive_toeplitz_matrix_seed) {
		zmq_disconnect(socket_seed_in, address_seed_in.c_str());
		zmq_close(socket_seed_in);
		zmq_ctx_destroy(socket_seed_in);
	}

	if (use_key_server)
	{
		zmq_disconnect(socket_key_in, address_key_in.c_str());
		zmq_close(socket_key_in);
		zmq_ctx_destroy(socket_key_in);
	}
}

string toHexString(const uint8_t* data, uint32_t data_length) {
	std::stringstream ss;
	ss << "{ ";
	for (int i = 0; i < data_length; ++i) {
		ss << std::uppercase << std::hex  << "0x" << std::setw(2) << std::setfill('0') << (int)data[i];
		(i % 8 == 7 && i+1 < data_length)
			? ss << "," << std::endl << "  "
			: ss << ", ";
	}
	ss.seekp(-2, std::ios_base::end);
	ss << " };";
	return ss.str();
}

bool isSha3(const uint8_t* dataToVerify, uint32_t dataToVerify_length, const uint8_t expectedHash[]) {
	sha3_ctx sha3;
	rhash_sha3_256_init(&sha3);
	rhash_sha3_update(&sha3, dataToVerify, dataToVerify_length);
	uint8_t* calculatedHash = (uint8_t*)malloc(32);
	rhash_sha3_final(&sha3, calculatedHash);
	//println(toHexString(calculatedHash, 32));
	return memcmp(calculatedHash, expectedHash, 32) == 0;
}

void verifyData(const uint8_t* dataToVerify) {
	if (isSha3(dataToVerify, vertical_len / 8, ampout_sha3)) {
		println("VERIFIED!");
	}
	else
	{
		println("VERIFICATION FAILED!");
		exit(101);
	}
}


void sendData() {
	int32_t rc;
	char syn[3];
	void* amp_out_socket = nullptr;
	if (host_ampout_server)
	{
		void* amp_out_context = zmq_ctx_new();
		amp_out_socket = zmq_socket(amp_out_context, ZMQ_REP);
		while (zmq_bind(amp_out_socket, address_amp_out.c_str()) != 0) {
			println("Binding to \"" << address_amp_out << "\" failed! Retrying...");
		}
	}

	int32_t ampOutsToStore = store_first_ampouts_in_file;
	fstream ampout_file;
	if (ampOutsToStore != 0) {
		ampout_file = fstream("ampout.bin", ios::out | ios::binary);
	}

	ThreadPool* verifyDataPool = nullptr;
	if (verify_ampout)
	{
		verifyDataPool = new ThreadPool(verify_ampout_threads);
	}
	auto start = chrono::high_resolution_clock::now();
	auto stop = chrono::high_resolution_clock::now();

	while (true) {

		while ((output_cache_read_pos + 1) % output_blocks_to_cache == output_cache_write_pos) {
			this_thread::yield();
		}
		output_cache_read_pos = (output_cache_read_pos + 1) % output_blocks_to_cache;

		uint8_t* output_block = testMemoryHost;

		if (verify_ampout)
		{
			verifyDataPool->enqueue(verifyData, output_block);
		}

		if (ampOutsToStore != 0) {
			if (ampOutsToStore > 0) {
				--ampOutsToStore;
			}
			ampout_file.write((char*)&output_block[0], vertical_len / 8);
			ampout_file.flush();
			if (ampOutsToStore == 0) {
				ampout_file.close();
			}
		}

		if (host_ampout_server)
		{
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
			if (show_zeromq_status) {
				println("Block sent to AMPOUT Client");
			}
		}

		stop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(stop - start).count();
		start = chrono::high_resolution_clock::now();

		if (show_ampout >= 0)
		{
			printlock.lock();
			cout << "Blocktime: " << duration / 1000.0 << " ms => " << (1000000.0 / duration) * (sample_size / 1000000.0) << " Mbit/s" << endl;
			if (show_ampout > 0)
			{
				for (size_t i = 0; i < min_template(vertical_block * sizeof(uint32_t), show_ampout); ++i)
				{
					printf("0x%02X: %s\n", output_block[i], bitset<8>(output_block[i]).to_string().c_str());
				}
			}
			fflush(stdout);
			printlock.unlock();
		}
	}
}


void readConfig() {
	Yaml::Node root;
	cout << "#Reading config.yaml..." << endl;
	try
	{
		Yaml::Parse(root, "config.yaml");
	}
	catch (const Yaml::Exception e)
	{
		cout << "Exception " << e.Type() << ": " << e.what() << endl;
		cout << "Can't open file config.yaml => terminating!" << endl;
		exit(102);
	}

	//45555 =>seed_in_alice, 46666 => seed_in_bob
	address_seed_in = root["address_seed_in"].As<string>("tcp://127.0.0.1:45555");
	address_key_in = root["address_key_in"].As<string>("tcp://127.0.0.1:47777");  //key_in
	address_amp_out = root["address_amp_out"].As<string>("tcp://127.0.0.1:48888"); //amp_out

	sample_size = static_cast<int>(round(pow(2, root["factor_exp"].As<uint32_t>(27))));
	reduction = static_cast<int>(round(pow(2, root["reduction_exp"].As<uint32_t>(11))));
	pre_mul_reduction = static_cast<int>(round(pow(2, root["pre_mul_reduction_exp"].As<uint32_t>(5))));
	cuda_device_id_to_use = root["cuda_device_id_to_use"].As<uint32_t>(1);
	input_blocks_to_cache = root["input_blocks_to_cache"].As<uint32_t>(16); //Has to be larger then 1
	output_blocks_to_cache = root["output_blocks_to_cache"].As<uint32_t>(16); //Has to be larger then 1

	dynamic_toeplitz_matrix_seed = root["dynamic_toeplitz_matrix_seed"].As<bool>(true);
	show_ampout = root["show_ampout"].As<int32_t>(0);
	show_zeromq_status = root["show_zeromq_status"].As<bool>(true);
	use_matrix_seed_server = root["use_matrix_seed_server"].As<bool>(true);
	use_key_server = root["use_key_server"].As<bool>(true);
	host_ampout_server = root["host_ampout_server"].As<bool>(true);
	store_first_ampouts_in_file = root["store_first_ampouts_in_file"].As<int32_t>(true);

	toeplitz_seed_path = root["toeplitz_seed_path"].As<string>("toeplitz_seed.bin");
	keyfile_path = root["keyfile_path"].As<string>("keyfile.bin");

	verify_ampout = root["verify_ampout"].As<bool>(true);
	verify_ampout_threads = root["verify_ampout_threads"].As<uint32_t>(8);


	vertical_len = sample_size / 4 + sample_size / 8;
	horizontal_len = sample_size / 2 + sample_size / 8;
	vertical_block = vertical_len / 32;
	horizontal_block = horizontal_len / 32;
	desired_block = sample_size / 32;
	key_blocks = desired_block + 1;
	input_cache_block_size = desired_block;
	output_cache_block_size = (desired_block + 31) * sizeof(uint32_t);
	recv_key = (uint32_t*)malloc(key_blocks * sizeof(uint32_t));
	key_start_zero_pos = (uint32_t*)malloc(input_blocks_to_cache * sizeof(uint32_t));
	key_rest_zero_pos = (uint32_t*)malloc(input_blocks_to_cache * sizeof(uint32_t));
}


inline void setConsoleDesign() {
	#ifdef _WIN32
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	DWORD dwConSize;
	COORD coordScreen = { 0, 0 };
	DWORD cCharsWritten;
	GetConsoleScreenBufferInfo(hConsole, &csbi);
	dwConSize = csbi.dwSize.X * csbi.dwSize.Y;
	FillConsoleOutputAttribute(hConsole,
		FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY | BACKGROUND_BLUE,
		dwConSize, coordScreen, &cCharsWritten);
	SetConsoleTextAttribute(hConsole,
		FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY | BACKGROUND_BLUE);
	#endif
}


#define PLAN_CUFFT \
if (cuFFT_planned) \
{ \
	/*Delete CUFFT Plans*/ \
	cufftDestroy(plan_forward_R2C); \
	cufftDestroy(plan_inverse_C2R); \
} \
\
/*Plan of the forward real to complex fast fourier transformation*/ \
cufftResult result_forward_FFT = cufftPlan1d(&plan_forward_R2C, sample_size, CUFFT_R2C, 1); \
if (result_forward_FFT != CUFFT_SUCCESS) \
{ \
	println("Failed to plan FFT 1! Error Code: " << result_forward_FFT); \
	exit(0); \
} \
\
/* Plan of the inverse complex to real fast fourier transformation */ \
cufftResult result_inverse_FFT = cufftPlan1d(&plan_inverse_C2R, sample_size, CUFFT_C2R, 1); \
if (result_inverse_FFT != CUFFT_SUCCESS) \
{ \
	println("Failed to plan IFFT 1! Error Code: " << result_inverse_FFT); \
	exit(0); \
} \
cuFFT_planned = true;


#if !defined(__NVCC__)
inline void VkFFTCreateConfiguration(VkGPU* vkGPU, vuda::detail::logical_device* logical_device, float* vkBuffer, VkFFTConfiguration* configuration)
{
	configuration->FFTdim = 1;
	configuration->size[0] = sample_size;
	configuration->size[1] = 1;
	configuration->size[2] = 1;
	configuration->performR2C = true;
	configuration->aimThreads = 1024;
	configuration->useLUT = false;
	configuration->normalize = false;
	configuration->device = &vkGPU->device;
	configuration->queue = &vkGPU->queue;
	configuration->fence = &vkGPU->fence;
	configuration->buffer = new VkBuffer{ logical_device->GetBuffer(vkBuffer) };
	bufferSize = (uint64_t)sizeof(float) * 2 * (sample_size / 2 + 1);
	configuration->bufferSize = &bufferSize;
	configuration->commandPool = &vkGPU->commandPool;
	configuration->physicalDevice = &vkGPU->physicalDevice;
	configuration->isCompilerInitialized = 1;
}


inline void planVkFFT(VkGPU* vkGPU, vuda::detail::logical_device* logical_device, VkFFTApplication* plan_forward_R2C_key, VkFFTApplication* plan_forward_R2C_seed, VkFFTApplication* plan_inverse_C2R, float* key_buffer, float* seed_buffer)
{
	if (cuFFT_planned)
	{
		/*Delete CUFFT Plans*/
		deleteVkFFT(plan_forward_R2C_seed);
		deleteVkFFT(plan_forward_R2C_key);
		deleteVkFFT(plan_inverse_C2R);
	}
	
	/*Plan of the forward real to complex fast fourier transformation*/
	VkFFTConfiguration plan_forward_R2C_key_configuration = {};
	VkFFTCreateConfiguration(vkGPU, logical_device, key_buffer, &plan_forward_R2C_key_configuration);
	plan_forward_R2C_key_configuration.performZeropadding[0] = true;
	plan_forward_R2C_key_configuration.fft_zeropad_left[0] = (plan_forward_R2C_key_configuration.size[0] / 4) + (plan_forward_R2C_key_configuration.size[0] / 16);
	plan_forward_R2C_key_configuration.fft_zeropad_right[0] = plan_forward_R2C_key_configuration.size[0];
	VkFFTResult result_forward_FFT_key = initializeVkFFT(plan_forward_R2C_key, plan_forward_R2C_key_configuration);
	if (result_forward_FFT_key != VKFFT_SUCCESS)
	{
		println("Failed to plan FFT key! Error Code: " << result_forward_FFT_key);
		exit(0);
	}

	/*Plan of the forward real to complex fast fourier transformation*/
	VkFFTConfiguration plan_forward_R2C_seed_configuration = {};
	VkFFTCreateConfiguration(vkGPU, logical_device, seed_buffer, &plan_forward_R2C_seed_configuration);
	VkFFTResult result_forward_FFT_seed = initializeVkFFT(plan_forward_R2C_seed, plan_forward_R2C_seed_configuration);
	if (result_forward_FFT_seed != VKFFT_SUCCESS)
	{
		println("Failed to plan FFT seed! Error Code: " << result_forward_FFT_seed);
		exit(0);
	}

	/*Plan of the forward real to complex fast fourier transformation*/
	VkFFTConfiguration plan_inverse_C2R_configuration = {};
	VkFFTCreateConfiguration(vkGPU, logical_device, key_buffer, &plan_inverse_C2R_configuration);
	VkFFTResult result_plan_inverse_C2R = initializeVkFFT(plan_inverse_C2R, plan_inverse_C2R_configuration);
	if (result_plan_inverse_C2R != VKFFT_SUCCESS)
	{
		println("Failed to plan IFFT! Error Code: " << result_plan_inverse_C2R);
		exit(0);
	}

	cuFFT_planned = true;
}
#endif


int main(int argc, char* argv[])
{
	//About
	#if defined(__NVCC__)
	string about = streamToString("# PrivacyAmplificationCuda v" << VERSION << " by Nico Bosshard from " << __DATE__ << " #");
	#else
	string about = streamToString("# PrivacyAmplification v" << VERSION << " by Nico Bosshard from " << __DATE__ << " #");
	#endif
	string border(about.length(), '#');
	cout << border << endl << about << endl << border << endl << endl;

	readConfig();

	cout << "#PrivacyAmplification with " << sample_size << " bits" << endl << endl;
	setConsoleDesign();

	cudaSetDevice(0); //cudaSetDevice(cuda_device_id_to_use);

	input_cache_read_pos = input_blocks_to_cache - 1;
	input_cache_write_pos = 0;
	output_cache_read_pos = input_blocks_to_cache - 1;
	output_cache_write_pos = 0;

	uint32_t* count_one_of_global_seed;
	uint32_t* count_one_of_global_key;
	float* correction_float_dev;
	Real* di1; //Device Input 1
	Real* di2; //Device Input 2
	Real* invOut;  //Result of the IFFT (uses the same memory as do2)
	#if defined(__NVCC__)
	Complex* do1;  //Device Output 1 and result of ElementWiseProduct
	Complex* do2;  //Device Output 2 and result of the IFFT
	#endif


	#if STOPWATCH == TRUE
	uint64_t stopwatch_wait_for_input_buffer = 0;
	uint64_t stopwatch_cleaned_memory = 0;
	uint64_t stopwatch_set_count_one_of_global_key_to_zero = 0;
	uint64_t stopwatch_set_count_one_of_global_seed_to_zero = 0;
	uint64_t stopwatch_binInt2float_key = 0;
	uint32_t stopwatch_binInt2float_seed = 0;
	uint64_t stopwatch_calculateCorrectionFloat = 0;
	uint64_t stopwatch_fft_key = 0;
	uint64_t stopwatch_fft_seed = 0;
	uint64_t stopwatch_setFirstElementToZero = 0;
	uint64_t stopwatch_elementWiseProduct = 0;
	uint64_t stopwatch_ifft = 0;
	uint64_t stopwatch_wait_for_output_buffer = 0;
	uint64_t stopwatch_toBinaryArray = 0;
	uint64_t stopwatch_total = 0;
	uint64_t stopwatch_total_max = UINT64_MAX;
	#endif

	#if defined(__NVCC__)
	cudaStream_t FFTStream, BinInt2floatKeyStream, BinInt2floatSeedStream, CalculateCorrectionFloatStream,
		cpu2gpuKeyStartStream, cpu2gpuKeyRestStream, cpu2gpuSeedStream, gpu2cpuStream,
		ElementWiseProductStream, ToBinaryArrayStream;
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
	#else
	const int cudaStream_t = 0;
	const int FFTStream = 0;
	const int BinInt2floatKeyStream = 0;
	const int BinInt2floatSeedStream = 0;
	const int CalculateCorrectionFloatStream = 0;
	const int cpu2gpuKeyStartStream = 0;
	const int cpu2gpuKeyRestStream = 0;
	const int cpu2gpuSeedStream = 0;
	const int gpu2cpuStream = 0;
	const int ElementWiseProductStream = 0;
	const int ToBinaryArrayStream = 0;

	VkGPU vkGPU = {};
	vkGPU.device_id = 0;
	cudaSetDevice(vkGPU.device_id);
	vkGPU.instance = vuda::detail::Instance::GetVkInstance();
	vkGPU.physicalDevice = vuda::detail::Instance::GetPhysicalDevice(vkGPU.device_id);
	vuda::detail::logical_device* logical_device = vuda::detail::interface_logical_devices::create(vkGPU.physicalDevice, 0);
	const vuda::detail::thrdcmdpool* thrdcmdpool = logical_device->GetPool(std::this_thread::get_id());
	vkGPU.device = logical_device->GetDeviceHandle();
	vkGPU.commandPool = thrdcmdpool->GetCommandPool();
	vkGPU.queue = logical_device->GetQueue(0);
	vkGPU.fence = thrdcmdpool->GetFence(0);
	VkFFTResult resFFT;

	#if(VKFFT_BACKEND==0)
	glslang_initialize_process();
	#elif(VKFFT_BACKEND==1)
	CUresult res = CUDA_SUCCESS;
	cudaError_t res2 = cudaSuccess;
	res = cuInit(0);
	if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	res2 = cudaSetDevice(vkGPU->device_id);
	if (res2 != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
	res = cuDeviceGet(&vkGPU->device, vkGPU->device_id);
	if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
	res = cuCtxCreate(&vkGPU->context, 0, vkGPU->device);
	if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
	#endif
	#endif

	// Allocate host pinned memory on RAM
	cudaMallocHost((void**)&toeplitz_seed, input_cache_block_size * sizeof(uint32_t) * input_blocks_to_cache);
	cudaMallocHost((void**)&key_start, input_cache_block_size * sizeof(uint32_t) * input_blocks_to_cache);
	cudaMallocHost((void**)&key_rest, input_cache_block_size * sizeof(uint32_t) * input_blocks_to_cache + 31 * sizeof(uint32_t));
	cudaMallocHost((void**)&Output, output_cache_block_size * output_blocks_to_cache);
	cudaMallocHost((void**)&assertKernelValue, sizeof(uint32_t));
	cudaMallocHost((void**)&assertKernelReturnValue, sizeof(uint32_t));
	cudaMallocHost((void**)&value_dev, sizeof(uint8_t));
	//#ifdef TEST
	cudaMallocHost((void**)&testMemoryHost, max(sample_size * sizeof(Complex), (sample_size + 992) * sizeof(Real)));
	//#endif
	#if SHOW_DEBUG_OUTPUT == TRUE
	cudaMallocHost((void**)&OutputFloat, sample_size * sizeof(float) * output_blocks_to_cache);
	#endif

	//Set key_start_zero_pos and key_rest_zero_pos to their default values
	fill(key_start_zero_pos, key_start_zero_pos + input_blocks_to_cache, desired_block);
	fill(key_rest_zero_pos, key_rest_zero_pos + input_blocks_to_cache, desired_block);

	// Allocate memory on GPU
	cudaMalloc((void**)&count_one_of_global_seed, sizeof(uint32_t));
	cudaMalloc((void**)&count_one_of_global_key, sizeof(uint32_t));
	cudaMallocHost((void**)&correction_float_dev, sizeof(float));

	cudaCalloc((void**)&di1, (uint64_t)sizeof(float) * 2 * ((sample_size + 992) / 2 + 1));
	
	/*Toeplitz matrix seed FFT input but this memory region is shared with invOut
	  if toeplitz matrix seed recalculation is disabled for the next block*/
	cudaMalloc((void**)&di2, (sample_size + 992) * sizeof(Real));

	#if defined(__NVCC__)
	/*Key FFT output but this memory region is shared with ElementWiseProduct output as they never conflict*/
	cudaMalloc((void**)&do1, sample_size * sizeof(Complex));

	/*Toeplitz Seed FFT output but this memory region is shared with invOut
	  if toeplitz matrix seed recalculation is enabled for the next block (default)*/
	cudaMalloc((void**)&do2, max(sample_size * sizeof(Complex), (sample_size + 992) * sizeof(Real)));
	#endif

	const uint32_t total_reduction = reduction * pre_mul_reduction;
	normalisation_float = ((float)sample_size) / ((float)total_reduction) / ((float)total_reduction);
	const Real float0 = 0.0f;
	const Real float1_reduced = 1.0f / reduction;
	#if defined(__NVCC__)
	const Complex complex0 = make_float2(0.0f, 0.0f);

	/*Copy constant variables from RAM to GPUs constant memory*/
	cudaMemcpyToSymbol(c0_dev, &complex0, sizeof(Complex));
	cudaMemcpyToSymbol(h0_dev, &float0, sizeof(float));
	cudaMemcpyToSymbol(h1_reduced_dev, &float1_reduced, sizeof(float));
	cudaMemcpyToSymbol(normalisation_float_dev, &normalisation_float, sizeof(float));
	cudaMemcpyToSymbol(sample_size_dev, &sample_size, sizeof(uint32_t));
	cudaMemcpyToSymbol(pre_mul_reduction_dev, &pre_mul_reduction, sizeof(uint32_t));
	#else
	float* float1_reduced_dev;
	cudaMallocHost((void**)&float1_reduced_dev, sizeof(float));
	*float1_reduced_dev = float1_reduced;

	float* normalisation_float_dev;
	cudaMallocHost((void**)&normalisation_float_dev, sizeof(float));
	*normalisation_float_dev = normalisation_float;

	uint32_t* sample_size_dev;
	cudaMallocHost((void**)&sample_size_dev, sizeof(uint32_t));
	*sample_size_dev = sample_size;

	uint32_t* pre_mul_reduction_dev;
	cudaMallocHost((void**)&pre_mul_reduction_dev, sizeof(uint32_t));
	*pre_mul_reduction_dev = pre_mul_reduction;
	#endif

	/*The reciveData function is parallelly executed on a separate thread which we start now*/
	thread threadReciveObj(reciveData);
	threadReciveObj.detach();

	/*The sendData function is parallelly executed on a separate thread which we start now*/
	thread threadSendObj(sendData);
	threadSendObj.detach();

	/*relevant_keyBlocks variables are used to detect dirty memory regions*/
	uint32_t relevant_keyBlocks = horizontal_block + 1;
	uint32_t relevant_keyBlocks_old = 0;

	bool recalculate_toeplitz_matrix_seed = true;
	bool speedtest = false;
	bool doTest = true;
	uint32_t dist_freq = sample_size / 2 + 1;

	#if defined(__NVCC__)
	/*Plan fast fourier transformations*/
	cufftHandle plan_forward_R2C;
	cufftHandle plan_inverse_C2R;
	PLAN_CUFFT;
	#else
	/*Plan fast fourier transformations*/
	VkFFTApplication plan_forward_R2C_seed = {};
	VkFFTApplication plan_forward_R2C_key = {};
	VkFFTApplication plan_inverse_C2R = {};
	planVkFFT(&vkGPU, logical_device, &plan_forward_R2C_key, &plan_forward_R2C_seed, &plan_inverse_C2R, di1, di2);
	#endif
	
	//unitTestCalculateCorrectionFloat();
	//unitTestSetFirstElementToZero();
	//unitTestElementWiseProduct();
	//unitTestBinInt2float();
	//unitTestToBinaryArray();
	
	for (char** arg = argv; *arg; ++arg) {
		if (strcmp(*arg, "speedtest") == 0) {
			speedtest = true;
			doTest = false;
			verify_ampout = false;
			host_ampout_server = false;
			auto start = chrono::high_resolution_clock::now();
			auto stop = chrono::high_resolution_clock::now();

			for (int i = 0; i < 2; ++i) {
				switch (i) {
					case 0: dynamic_toeplitz_matrix_seed = true; break;
					case 1: dynamic_toeplitz_matrix_seed = false; break;
				}
				for (int j = 10; j < 28; ++j) {
					sample_size = pow(2, j);
					vertical_len = sample_size / 4 + sample_size / 8;
					horizontal_len = sample_size / 2 + sample_size / 8;
					vertical_block = vertical_len / 32;
					horizontal_block = horizontal_len / 32;
					desired_block = sample_size / 32;
					key_blocks = desired_block + 1;
					normalisation_float = ((float)sample_size) / ((float)total_reduction) / ((float)total_reduction);
					dist_freq = sample_size / 2 + 1;
					#if defined(__NVCC__)
					cudaMemcpyToSymbol(normalisation_float_dev, &normalisation_float, sizeof(float));
					cudaMemcpyToSymbol(sample_size_dev, &sample_size, sizeof(uint32_t));
					PLAN_CUFFT;
					#else
					*normalisation_float_dev = normalisation_float;
					*sample_size_dev = sample_size;
					planVkFFT(&vkGPU, logical_device, &plan_forward_R2C_seed, &plan_forward_R2C_key, &plan_inverse_C2R, di1, di2);
					#endif
					for (int k = 0; k < 10; ++k) {
						//GOSUB reimplementation - Function call in same stackframe
						goto mainloop;
						return_speedtest:;

						stop = chrono::high_resolution_clock::now();
						auto duration = chrono::duration_cast<chrono::microseconds>(stop - start).count();
						start = chrono::high_resolution_clock::now();
						println("d[" << i << "," << j << "," << k << "]=" << (1000000.0 / duration) * (sample_size / 1000000.0));
					}
				}
			}
			exit(0);
		}
		if (strncmp(*arg, "unitTest", 8) != 0) continue;
		if (strcmp(*arg, "unitTestCalculateCorrectionFloat") == 0) exit(unitTestCalculateCorrectionFloat());
		if (strcmp(*arg, "unitTestSetFirstElementToZero") == 0) exit(unitTestSetFirstElementToZero());
		if (strcmp(*arg, "unitTestElementWiseProduct") == 0) exit(unitTestElementWiseProduct());
		if (strcmp(*arg, "unitTestBinInt2float") == 0) exit(unitTestBinInt2float());
		if (strcmp(*arg, "unitTestToBinaryArray") == 0) exit(unitTestToBinaryArray());
	}



	//##########################
	// Mainloop of main thread #
	//##########################
	while (true) {
		mainloop:;
		STOPWATCH_START
		/*Spinlock waiting for data provider*/
		chrono::steady_clock::time_point begin = std::chrono::high_resolution_clock::now();
		while ((input_cache_read_pos + 1) % input_blocks_to_cache == input_cache_write_pos) {
			this_thread::yield();
		}
		input_cache_read_pos = (input_cache_read_pos + 1) % input_blocks_to_cache; //Switch read cache
		STOPWATCH_SAVE(stopwatch_wait_for_input_buffer)

		#if defined(__NVCC__)
		/*Detect dirty memory regions parts*/
		/*Not needed on VkFFT as we can make use of it's native zero padding instead*/
		relevant_keyBlocks_old = relevant_keyBlocks;
		relevant_keyBlocks = horizontal_block + 1;
		if (relevant_keyBlocks_old > relevant_keyBlocks) {
			/*Fill dirty memory regions parts with zeros*/
			cudaMemset(di1 + relevant_keyBlocks, 0b00000000, (relevant_keyBlocks_old - relevant_keyBlocks) * sizeof(Real));
		}
		STOPWATCH_SAVE(stopwatch_cleaned_memory)
		#endif

		cudaMemset(count_one_of_global_key, 0b00000000, sizeof(uint32_t));
		#ifdef TEST
		if (doTest) {
			assertGPU(count_one_of_global_key, 1, 0);
			assertTrue(isSha3(reinterpret_cast<uint8_t*>(key_start + input_cache_block_size * input_cache_read_pos), relevant_keyBlocks * sizeof(uint32_t), binInt2float_key_binIn_hash));
		}
		#endif
		STOPWATCH_SAVE(stopwatch_set_count_one_of_global_key_to_zero)
		#if defined(__NVCC__)
		binInt2float KERNEL_ARG4((int)((relevant_keyBlocks * 32 + 1023) / 1024), min_template(relevant_keyBlocks * 32, 1024), 0,
			BinInt2floatKeyStream) (key_start + input_cache_block_size * input_cache_read_pos, di1, count_one_of_global_key);
		#else
		vuda::launchKernel("SPIRV/binInt2float.spv", "main", BinInt2floatKeyStream, (int)((relevant_keyBlocks * 32 + 1023) / 1024), min_template(relevant_keyBlocks * 32, 1024), key_start + input_cache_block_size * input_cache_read_pos, di1, count_one_of_global_key, float1_reduced_dev);
		#endif
		cudaStreamSynchronize(BinInt2floatKeyStream);
		#ifdef TEST
		if (doTest) {
			cudaMemcpy(testMemoryHost, di1, relevant_keyBlocks * 32 * sizeof(Real), cudaMemcpyDeviceToHost);
			assertTrue(isSha3(const_cast<uint8_t*>(testMemoryHost), relevant_keyBlocks * 32 * sizeof(Real), binInt2float_key_floatOut_hash));
		}
		#endif
		STOPWATCH_SAVE(stopwatch_binInt2float_key)
		if (recalculate_toeplitz_matrix_seed) {
			cudaMemset(count_one_of_global_seed, 0x00, sizeof(uint32_t));
			#ifdef TEST
			if (doTest) {
				assertGPU(count_one_of_global_seed, 1, 0);
				assertTrue(isSha3(reinterpret_cast<uint8_t*>(toeplitz_seed + input_cache_block_size * input_cache_read_pos), desired_block * sizeof(uint32_t), binInt2float_seed_binIn_hash));
			}
			#endif
			STOPWATCH_SAVE(stopwatch_set_count_one_of_global_seed_to_zero)
			#if defined(__NVCC__)
			binInt2float KERNEL_ARG4((int)(((int)(sample_size)+1023) / 1024), min_template(sample_size, 1024), 0,
				BinInt2floatSeedStream) (toeplitz_seed + input_cache_block_size * input_cache_read_pos, di2, count_one_of_global_seed);
			#else
			vuda::launchKernel("SPIRV/binInt2float.spv", "main", BinInt2floatSeedStream, (int)(((int)(sample_size)+1023) / 1024), min_template(sample_size, 1024), toeplitz_seed + input_cache_block_size * input_cache_read_pos, di2, count_one_of_global_seed, float1_reduced_dev);
			#endif
			cudaStreamSynchronize(BinInt2floatSeedStream);
			#ifdef TEST
			if (doTest) {
				cudaMemcpy(testMemoryHost, di2, sample_size * sizeof(Real), cudaMemcpyDeviceToHost);
				assertTrue(isSha3(const_cast<uint8_t*>(testMemoryHost), sample_size * sizeof(Real), binInt2float_seed_floatOut_hash));
			}
			#endif
			STOPWATCH_SAVE(stopwatch_binInt2float_seed)
		}
		
		#ifdef TEST
		if (doTest) {
			assertGPU(count_one_of_global_key, 1, 41947248);
			assertGPU(count_one_of_global_seed, 1, 67113455);
		}
		#endif
		#if defined(__NVCC__)
		calculateCorrectionFloat KERNEL_ARG4(1, 1, 0, CalculateCorrectionFloatStream)
			(count_one_of_global_key, count_one_of_global_seed, correction_float_dev);
		#else
		vuda::launchKernel("SPIRV/calculateCorrectionFloat.spv", "main", CalculateCorrectionFloatStream, 1, 1, count_one_of_global_key, count_one_of_global_seed, correction_float_dev, sample_size_dev);
		#endif
		cudaStreamSynchronize(CalculateCorrectionFloatStream);
		STOPWATCH_SAVE(stopwatch_calculateCorrectionFloat)
		#if defined(__NVCC__)
		cufftExecR2C(plan_forward_R2C, di1, do1);
		cudaDeviceSynchronize();
		STOPWATCH_SAVE(stopwatch_fft_key)
		
		if (recalculate_toeplitz_matrix_seed) {
			cufftExecR2C(plan_forward_R2C, di2, do2);
			if (!dynamic_toeplitz_matrix_seed)
			{
				recalculate_toeplitz_matrix_seed = false;
			}
			cudaDeviceSynchronize();
			STOPWATCH_SAVE(stopwatch_fft_seed)
		}
		Complex* intermediate_key = reinterpret_cast<Complex*>(do1);
		Complex* intermediate_seed = reinterpret_cast<Complex*>(do2);
		invOut = reinterpret_cast<Real*>(di2); //invOut and di2 share together the same memory region
		#else
		vkfftExecR2C(&vkGPU, &plan_forward_R2C_key);
		cudaStreamSynchronize(FFTStream);
		STOPWATCH_SAVE(stopwatch_fft_key)
		if (recalculate_toeplitz_matrix_seed) {
			vkfftExecR2C(&vkGPU, &plan_forward_R2C_seed);
			if (!dynamic_toeplitz_matrix_seed)
			{
				recalculate_toeplitz_matrix_seed = false;
			}
			cudaStreamSynchronize(FFTStream);
			STOPWATCH_SAVE(stopwatch_fft_seed)
		}
		Complex* intermediate_key = reinterpret_cast<Complex*>(di1);
		Complex* intermediate_seed = reinterpret_cast<Complex*>(di2);
		invOut = reinterpret_cast<Real*>(di1); //invOut and di2 share together the same memory region
		#endif
		#ifdef TEST
		if (doTest) {
			cudaMemcpy(testMemoryHost, intermediate_key, 2 * (sample_size / 2 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
			for (int i = 0; i < 100; i += 2) {
				println(i << ": " << reinterpret_cast<float*>(testMemoryHost)[i] << "|" << reinterpret_cast<float*>(testMemoryHost)[i + 1]);
			}
			for (int i = sample_size - 50; i < sample_size + 50; i += 2) {
				println(i << ": " << reinterpret_cast<float*>(testMemoryHost)[i] << "|" << reinterpret_cast<float*>(testMemoryHost)[i + 1]);
			}
			assertTrue(isFletcherFloat(reinterpret_cast<float*>(testMemoryHost), 2 * (sample_size / 2 + 1), 169418278.63041568, 200.0, 11374845421549196.0, 20000000000.0));
			cudaMemcpy(testMemoryHost, intermediate_seed, 2 * (sample_size / 2 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
			assertTrue(isFletcherFloat(reinterpret_cast<float*>(testMemoryHost), 2 * (sample_size / 2 + 1), 214211928.23554835, 200.0, 14378010673396208.0, 20000000000.0));
		}
		#endif
		#if defined(__NVCC__)
		setFirstElementToZero KERNEL_ARG4(1, 2, 0, ElementWiseProductStream) (intermediate_key, intermediate_seed);
		#else
		vuda::launchKernel("SPIRV/setFirstElementToZero.spv", "main", ElementWiseProductStream, 1, 2, intermediate_key, intermediate_seed);
		#endif
		cudaStreamSynchronize(ElementWiseProductStream);
		#ifdef TEST
		if (doTest) {
			cudaMemcpy(testMemoryHost, intermediate_key, 2 * (sample_size / 2 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
			for (int i = 0; i < 100; i += 2) {
				println(i << ": " << reinterpret_cast<float*>(testMemoryHost)[i] << "|" << reinterpret_cast<float*>(testMemoryHost)[i + 1]);
			}
			for (int i = sample_size - 50; i < sample_size + 50; i += 2) {
				println(i << ": " << reinterpret_cast<float*>(testMemoryHost)[i] << "|" << reinterpret_cast<float*>(testMemoryHost)[i + 1]);
			}
			assertTrue(isFletcherFloat(reinterpret_cast<float*>(testMemoryHost), 2 * (sample_size / 2 + 1), 169397796.57572800, 200.0, 11372096366664388.0, 20000000000.0));
			cudaMemcpy(testMemoryHost, intermediate_seed, 2 * (sample_size / 2 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
			assertTrue(isFletcherFloat(reinterpret_cast<float*>(testMemoryHost), 2 * (sample_size / 2 + 1), 214179157.99336109, 200.0, 14373612325878530.0, 20000000000.0));
		}
		#endif
		STOPWATCH_SAVE(stopwatch_setFirstElementToZero)
		#if defined(__NVCC__)
		ElementWiseProduct KERNEL_ARG4((int)((dist_freq + 1023) / 1024), min((int)dist_freq, 1024), 0, ElementWiseProductStream) (intermediate_key, intermediate_seed);
		#else
		vuda::launchKernel("SPIRV/elementWiseProduct.spv", "main", ElementWiseProductStream, (int)((dist_freq + 1023) / 1024), min((int)dist_freq, 1024), intermediate_key, intermediate_seed, pre_mul_reduction_dev);
		#endif
		cudaStreamSynchronize(ElementWiseProductStream);
		#ifdef TEST
		if (doTest) {
			cudaMemcpy(testMemoryHost, intermediate_key, 2 * (sample_size / 2 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
			assertTrue(isFletcherFloat(reinterpret_cast<float*>(testMemoryHost), 2 * (sample_size / 2 + 1) * 2, 414613.13602233, 0.5, 83481560389295.703125, 200000000.0));
		}
		#endif
		STOPWATCH_SAVE(stopwatch_elementWiseProduct)
		#if defined(__NVCC__)
		cufftExecC2R(plan_inverse_C2R, intermediate_key, invOut);
		cudaDeviceSynchronize();
		#else
		vkfftExecC2R(&vkGPU, &plan_inverse_C2R);
		cudaStreamSynchronize(FFTStream);
		#endif
		STOPWATCH_SAVE(stopwatch_ifft)


		/*Spinlock waiting for the data consumer*/
		if (!speedtest) {
			while (output_cache_write_pos % output_blocks_to_cache == output_cache_read_pos) {
				this_thread::yield();
			}
			STOPWATCH_SAVE(stopwatch_wait_for_output_buffer)
		}

		/*Calculates where in the host pinned output memory the Privacy Amplification result will be stored*/
		uint32_t* binOut = reinterpret_cast<uint32_t*>(Output + output_cache_block_size * output_cache_write_pos);
		#ifdef TEST
		if (doTest) {
			cudaMemcpy(testMemoryHost, invOut, sample_size * sizeof(Real), cudaMemcpyDeviceToHost);
			for (int i = 0; i < 100; i += 2) {
				println(i << ": " << reinterpret_cast<float*>(testMemoryHost)[i] << "|" << reinterpret_cast<float*>(testMemoryHost)[i + 1]);
			}
			for (int i = sample_size - 50; i < sample_size + 50; i += 2) {
				println(i << ": " << reinterpret_cast<float*>(testMemoryHost)[i] << "|" << reinterpret_cast<float*>(testMemoryHost)[i + 1]);
			}
			for (uint32_t i = 0; i < 100; ++i) {
				printf("%f ", reinterpret_cast<float*>(testMemoryHost)[i] / normalisation_float + *correction_float_dev);
				if (i % 8 == 7) std::cout << "\n";
			}
			for (uint32_t i = sample_size - 50; i < sample_size + 50; ++i) {
				printf("%f ", reinterpret_cast<float*>(testMemoryHost)[i] / normalisation_float + *correction_float_dev);
				if (i % 8 == 7) std::cout << "\n";
			}
			println("");
			#if SHOW_DEBUG_OUTPUT == TRUE
			FILE* pFile;
			#if defined(__NVCC__)
			pFile = fopen("Result_Cuda.txt", "w");
			#else
			pFile = fopen("Result_Vulkan.txt", "w");
			#endif
			for (uint32_t i = 0; i < sample_size; ++i) {
				fprintf(pFile, "%i", (int)roundf(reinterpret_cast<float*>(testMemoryHost)[i] / normalisation_float + *correction_float_dev) & 1);
				if (i % 192 == 191) fprintf(pFile, "\n");
			}
			fclose(pFile);
			exit(0);
			#endif
			assertTrue(isFletcherFloat(reinterpret_cast<float*>(testMemoryHost), sample_size, 8112419221.92300797, 20000.0, 542186359506315456.0, 2000000000000.0));
			assertTrue(isSha3(reinterpret_cast<uint8_t*>(key_rest + input_cache_block_size * input_cache_read_pos), vertical_len / 8, key_rest_hash));
			assertGPU(reinterpret_cast<uint32_t*>(correction_float_dev), 1, 0x3F54D912); //0.83143723	
		}		
		#endif
		#if defined(__NVCC__)
		ToBinaryArray KERNEL_ARG4((int)((int)(vertical_block) / 31) + 1, 1023, 0, ToBinaryArrayStream)
			(invOut, reinterpret_cast<uint32_t*>(testMemoryHost), key_rest + input_cache_block_size * input_cache_read_pos, correction_float_dev);
		#else
		vuda::launchKernel("SPIRV/toBinaryArray.spv", "main", ToBinaryArrayStream, (int)((int)(vertical_block) / 31) + 1, 1023, invOut, testMemoryHost, key_rest + input_cache_block_size * input_cache_read_pos, correction_float_dev, normalisation_float_dev);
		#endif
		cudaStreamSynchronize(ToBinaryArrayStream);
		#ifdef TEST
		if (doTest) {
			assertTrue(isSha3(reinterpret_cast<uint8_t*>(testMemoryHost), vertical_len / 8, ampout_sha3));
		}
		#endif
		//printBin(reinterpret_cast<uint8_t*>(testMemoryHost), reinterpret_cast<uint8_t*>(Output + output_cache_block_size * output_cache_write_pos) + 200);
		STOPWATCH_SAVE(stopwatch_toBinaryArray)
		STOPWATCH_TOTAL(stopwatch_total)

			
		#if STOPWATCH == TRUE
		if (stopwatch_total < stopwatch_total_max) {
			stopwatch_total_max = stopwatch_total;
			println(fixed << setprecision(3) <<
					"wait_for_input_buffer    " << stopwatch_wait_for_input_buffer / 1000000.0 << " ms\n" <<
					"cleaned_memory           " << stopwatch_cleaned_memory / 1000000.0 << " ms\n" <<
					"set_count_key_to_zero    " << stopwatch_set_count_one_of_global_key_to_zero / 1000000.0 << " ms\n" <<
					"set_count_seed_to_zero   " << stopwatch_set_count_one_of_global_seed_to_zero / 1000000.0 << " ms\n" <<
					"binIntffloat_key         " << stopwatch_binInt2float_key / 1000000.0 << " ms\n" <<
					"binIntffloat_seed        " << stopwatch_binInt2float_seed / 1000000.0 << " ms\n" <<
					"calculateCorrectionFloat " << stopwatch_calculateCorrectionFloat / 1000000.0 << " ms\n" <<
					"fft_key                  " << stopwatch_fft_key / 1000000.0 << " ms\n" <<
					"fft_seed                 " << stopwatch_fft_seed / 1000000.0 << " ms\n" <<
					"setFirstElementToZero    " << stopwatch_setFirstElementToZero / 1000000.0 << " ms\n" <<
					"elementWiseProduct       " << stopwatch_elementWiseProduct / 1000000.0 << " ms\n" <<
					"ifft                     " << stopwatch_ifft / 1000000.0 << " ms\n" <<
					"wait_for_output_buffer   " << stopwatch_wait_for_output_buffer / 1000000.0 << " ms\n" <<
					"toBinaryArray            " << stopwatch_toBinaryArray / 1000000.0 << " ms\n" <<
					"Total                    " << stopwatch_total / 1000000.0 << " ms\n" <<
					"Speed                    " << (1000000000.0 / stopwatch_total) * (sample_size / 1000000.0) << " MBit/s");
		}
		#endif


		if (speedtest) {
			goto return_speedtest;
		}
		else
		{
			output_cache_write_pos = (output_cache_write_pos + 1) % output_blocks_to_cache;
		}

	}

	#if defined(__NVCC__)
	// Delete CUFFT Plans
	cufftDestroy(plan_forward_R2C);
	cufftDestroy(plan_inverse_C2R);
	#else
	// Delete CUFFT Plans
	deleteVkFFT(&plan_forward_R2C_seed);
	deleteVkFFT(&plan_forward_R2C_key);
	#endif

	// Deallocate memoriey on GPU and RAM
	cudaFree(di1);
	cudaFree(di2);
	cudaFree(invOut);
	#if defined(__NVCC__)
	cudaFree(do1);
	cudaFree(do2);
	#endif
	cudaFree(Output);
	cudaFree(testMemoryHost);
	return 0;
}
