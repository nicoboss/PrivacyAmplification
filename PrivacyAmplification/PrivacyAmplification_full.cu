//cpp -x c++ -I/usr/local/cuda-11.2/include/ PrivacyAmplification.cu PrivacyAmplification_full.cu
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
#include <thread>
#include <atomic>
#include <bitset>
#include <future>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <math.h>
#include "yaml/Yaml.hpp"
#include "sha3/sha3.h"
#include "ThreadPool.h"


       




typedef float Real;

typedef float2 Complex;
uint32_t sample_size;
uint32_t blocks_in_bank;




uint32_t reduction;




uint32_t pre_mul_reduction;



uint32_t cuda_device_id_to_use;




uint32_t input_banks_to_cache;




uint32_t output_banks_to_cache;






bool dynamic_toeplitz_matrix_seed;




int32_t show_ampout;


bool show_zeromq_status;
bool use_matrix_seed_server;
std::string toeplitz_seed_path;
bool use_key_server;
std::string keyfile_path;







bool host_ampout_server;



int32_t store_first_ampouts_in_file;
bool verify_ampout;





uint32_t verify_ampout_threads;
const uint8_t ampout_sha3[] = { 0xC4, 0x22, 0xB6, 0x86, 0x5C, 0x72, 0xCA, 0xD8,
                                0x2C, 0xC2, 0x6A, 0x14, 0x62, 0xB8, 0xA4, 0x56,
                                0x6F, 0x91, 0x17, 0x50, 0xF3, 0x1B, 0x14, 0x75,
                                0x69, 0x12, 0x69, 0xC1, 0xB7, 0xD4, 0xA7, 0x16 };

const uint8_t binInt2float_key_binIn_hash[] = { 0x64, 0xE3, 0x6A, 0xEC, 0xDA, 0xD6, 0x4F, 0x8E,
                                                0x16, 0x87, 0xD2, 0x70, 0x96, 0x63, 0xF8, 0xD0,
                                                0x17, 0xFA, 0xA8, 0x35, 0xD2, 0x8D, 0x9A, 0x2D,
                                                0xD3, 0x1F, 0x05, 0x13, 0xBD, 0x92, 0xFD, 0xB1 };

const uint8_t binInt2float_key_floatOut_hash[] = { 0x40, 0x36, 0xEC, 0xAA, 0xB8, 0x13, 0x59, 0x52,
                                                   0x59, 0x8E, 0xCA, 0x82, 0x39, 0x84, 0x4C, 0x44,
                                                   0xC3, 0x83, 0x30, 0x55, 0x83, 0xDB, 0x26, 0x25,
                                                   0xB2, 0xAC, 0x03, 0x1A, 0xC1, 0xCF, 0xEA, 0x50 };

const uint8_t binInt2float_seed_binIn_hash[] = { 0x06, 0x15, 0xDF, 0x04, 0xF7, 0xF8, 0x98, 0x8E,
                                                 0x31, 0xC1, 0x4F, 0xE9, 0x2A, 0xE9, 0x0E, 0xDE,
                                                 0xA1, 0x4F, 0x33, 0xE5, 0xD0, 0x89, 0xE3, 0xC5,
                                                 0x7F, 0xE6, 0x79, 0xFB, 0xA6, 0xFB, 0x35, 0x05 };

const uint8_t binInt2float_seed_floatOut_hash[] = { 0x71, 0x15, 0xFA, 0x9A, 0xDA, 0x0A, 0xD5, 0x34,
                                                    0x95, 0x33, 0xA5, 0x2B, 0x80, 0x4E, 0x2F, 0xF4,
                                                    0xE6, 0xF6, 0x82, 0x37, 0xF0, 0x84, 0x28, 0x97,
                                                    0x51, 0xF4, 0x87, 0xB7, 0x00, 0x49, 0x09, 0xAF };

const uint8_t cufftExecC2R_output_hash[] = { 0x95, 0x5C, 0x8F, 0x01, 0xB3, 0x3C, 0x9C, 0x52,
                                             0x0D, 0x48, 0xFC, 0xA9, 0x09, 0xFA, 0x0D, 0x77,
                                             0x6C, 0xBC, 0x86, 0xD5, 0x5A, 0x21, 0x59, 0xC7,
                                             0xAD, 0x45, 0x7C, 0x60, 0x83, 0xF4, 0xA7, 0x5F };

const uint8_t key_rest_hash[] = { 0x11, 0x8D, 0x08, 0x8C, 0xE2, 0xC3, 0xDA, 0x93,
                                  0x32, 0xB8, 0x91, 0x57, 0xA1, 0x8B, 0x03, 0x26,
                                  0xA0, 0x17, 0x9A, 0x1E, 0x65, 0xFC, 0xC5, 0xBE,
                                  0xD2, 0x01, 0xDF, 0xE6, 0xB8, 0x2A, 0xC4, 0xE2 };







void printStream(std::ostream& os);







void printlnStream(std::ostream& os);






std::string convertStreamToString(std::ostream& os);







 void cudaAssertValue(uint32_t * data);
 void calculateCorrectionFloat(uint32_t* count_one_of_global_seed, uint32_t* count_one_of_global_key, float* correction_float_dev);
int unitTestCalculateCorrectionFloat();
 void setFirstElementToZero(Complex* do1, Complex* do2);
int unitTestSetFirstElementToZero();
 void ElementWiseProduct(Complex* do1, Complex* do2);
int unitTestElementWiseProduct();

std::pair<double, double> FletcherFloat(float* data, int count);

bool isFletcherFloat(float* data, int count, const double expectedSum1, const double allowedAbsDeltaSum1, const double expectedSum2, const double allowedAbsDeltaSum2);







 void binInt2float(uint32_t* binIn, Real* realOut, uint32_t* count_one_global);
void unitTestBinInt2floatVerifyResultThread(float* floatOutTest, int i, int i_max);
int unitTestBinInt2float();

unsigned A000788(unsigned n);
 void ToBinaryArray(Real* invOut, uint32_t* binOut, uint32_t* key_rest_local, Real* correction_float_dev, uint32_t block_index_divisor);
void unitTestToBinaryArrayVerifyResultThread(uint32_t* binOutTest, uint32_t* key_rest_local, int i, int i_max);
int unitTestToBinaryArray();






void printBin(const uint8_t* position, const uint8_t* end);






void printBin(const uint32_t* position, const uint32_t* end);
void key2StartRest();






void readMatrixSeedFromFile();






void readKeyFromFile();
void reciveData();

std::string toHexString(const unsigned char* data, uint32_t data_length);

bool isSha3(const unsigned char* dataToVerify, uint32_t dataToVerifySize, const uint8_t expectedHash[]);
void verifyData(const unsigned char* dataToVerify);
void sendData();




void readConfig();


void setConsoleDesign();







int main(int argc, char* argv[]);


template< typename T >
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        fprintf(
               stderr
                     , "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        cudaDeviceReset();
        exit(
            1
                        );
    }
}
static const char* _cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";

    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "CUFFT_INCOMPLETE_PARAMETER_LIST";

    case CUFFT_INVALID_DEVICE:
        return "CUFFT_INVALID_DEVICE";

    case CUFFT_PARSE_ERROR:
        return "CUFFT_PARSE_ERROR";

    case CUFFT_NO_WORKSPACE:
        return "CUFFT_NO_WORKSPACE";

    case CUFFT_NOT_IMPLEMENTED:
        return "CUFFT_NOT_IMPLEMENTED";

    case CUFFT_LICENSE_ERROR:
        return "CUFFT_LICENSE_ERROR";
    }

    return "<unknown>";
}

static const char* _cudaGetErrorEnum(cudaError_t error)
{
    switch (error)
    {
    case cudaSuccess:
        return "cudaSuccess";

    case cudaErrorMissingConfiguration:
        return "cudaErrorMissingConfiguration";

    case cudaErrorMemoryAllocation:
        return "cudaErrorMemoryAllocation";

    case cudaErrorInitializationError:
        return "cudaErrorInitializationError";

    case cudaErrorLaunchFailure:
        return "cudaErrorLaunchFailure";

    case cudaErrorPriorLaunchFailure:
        return "cudaErrorPriorLaunchFailure";

    case cudaErrorLaunchTimeout:
        return "cudaErrorLaunchTimeout";

    case cudaErrorLaunchOutOfResources:
        return "cudaErrorLaunchOutOfResources";

    case cudaErrorInvalidDeviceFunction:
        return "cudaErrorInvalidDeviceFunction";

    case cudaErrorInvalidConfiguration:
        return "cudaErrorInvalidConfiguration";

    case cudaErrorInvalidDevice:
        return "cudaErrorInvalidDevice";

    case cudaErrorInvalidValue:
        return "cudaErrorInvalidValue";

    case cudaErrorInvalidPitchValue:
        return "cudaErrorInvalidPitchValue";

    case cudaErrorInvalidSymbol:
        return "cudaErrorInvalidSymbol";

    case cudaErrorMapBufferObjectFailed:
        return "cudaErrorMapBufferObjectFailed";

    case cudaErrorUnmapBufferObjectFailed:
        return "cudaErrorUnmapBufferObjectFailed";

    case cudaErrorInvalidHostPointer:
        return "cudaErrorInvalidHostPointer";

    case cudaErrorInvalidDevicePointer:
        return "cudaErrorInvalidDevicePointer";

    case cudaErrorInvalidTexture:
        return "cudaErrorInvalidTexture";

    case cudaErrorInvalidTextureBinding:
        return "cudaErrorInvalidTextureBinding";

    case cudaErrorInvalidChannelDescriptor:
        return "cudaErrorInvalidChannelDescriptor";

    case cudaErrorInvalidMemcpyDirection:
        return "cudaErrorInvalidMemcpyDirection";

    case cudaErrorAddressOfConstant:
        return "cudaErrorAddressOfConstant";

    case cudaErrorTextureFetchFailed:
        return "cudaErrorTextureFetchFailed";

    case cudaErrorTextureNotBound:
        return "cudaErrorTextureNotBound";

    case cudaErrorSynchronizationError:
        return "cudaErrorSynchronizationError";

    case cudaErrorInvalidFilterSetting:
        return "cudaErrorInvalidFilterSetting";

    case cudaErrorInvalidNormSetting:
        return "cudaErrorInvalidNormSetting";

    case cudaErrorMixedDeviceExecution:
        return "cudaErrorMixedDeviceExecution";

    case cudaErrorCudartUnloading:
        return "cudaErrorCudartUnloading";

    case cudaErrorUnknown:
        return "cudaErrorUnknown";

    case cudaErrorNotYetImplemented:
        return "cudaErrorNotYetImplemented";

    case cudaErrorMemoryValueTooLarge:
        return "cudaErrorMemoryValueTooLarge";

    case cudaErrorInvalidResourceHandle:
        return "cudaErrorInvalidResourceHandle";

    case cudaErrorNotReady:
        return "cudaErrorNotReady";

    case cudaErrorInsufficientDriver:
        return "cudaErrorInsufficientDriver";

    case cudaErrorSetOnActiveProcess:
        return "cudaErrorSetOnActiveProcess";

    case cudaErrorInvalidSurface:
        return "cudaErrorInvalidSurface";

    case cudaErrorNoDevice:
        return "cudaErrorNoDevice";

    case cudaErrorECCUncorrectable:
        return "cudaErrorECCUncorrectable";

    case cudaErrorSharedObjectSymbolNotFound:
        return "cudaErrorSharedObjectSymbolNotFound";

    case cudaErrorSharedObjectInitFailed:
        return "cudaErrorSharedObjectInitFailed";

    case cudaErrorUnsupportedLimit:
        return "cudaErrorUnsupportedLimit";

    case cudaErrorDuplicateVariableName:
        return "cudaErrorDuplicateVariableName";

    case cudaErrorDuplicateTextureName:
        return "cudaErrorDuplicateTextureName";

    case cudaErrorDuplicateSurfaceName:
        return "cudaErrorDuplicateSurfaceName";

    case cudaErrorDevicesUnavailable:
        return "cudaErrorDevicesUnavailable";

    case cudaErrorInvalidKernelImage:
        return "cudaErrorInvalidKernelImage";

    case cudaErrorNoKernelImageForDevice:
        return "cudaErrorNoKernelImageForDevice";

    case cudaErrorIncompatibleDriverContext:
        return "cudaErrorIncompatibleDriverContext";

    case cudaErrorPeerAccessAlreadyEnabled:
        return "cudaErrorPeerAccessAlreadyEnabled";

    case cudaErrorPeerAccessNotEnabled:
        return "cudaErrorPeerAccessNotEnabled";

    case cudaErrorDeviceAlreadyInUse:
        return "cudaErrorDeviceAlreadyInUse";

    case cudaErrorProfilerDisabled:
        return "cudaErrorProfilerDisabled";

    case cudaErrorProfilerNotInitialized:
        return "cudaErrorProfilerNotInitialized";

    case cudaErrorProfilerAlreadyStarted:
        return "cudaErrorProfilerAlreadyStarted";

    case cudaErrorProfilerAlreadyStopped:
        return "cudaErrorProfilerAlreadyStopped";


    case cudaErrorAssert:
        return "cudaErrorAssert";

    case cudaErrorTooManyPeers:
        return "cudaErrorTooManyPeers";

    case cudaErrorHostMemoryAlreadyRegistered:
        return "cudaErrorHostMemoryAlreadyRegistered";

    case cudaErrorHostMemoryNotRegistered:
        return "cudaErrorHostMemoryNotRegistered";


    case cudaErrorOperatingSystem:
        return "cudaErrorOperatingSystem";

    case cudaErrorPeerAccessUnsupported:
        return "cudaErrorPeerAccessUnsupported";

    case cudaErrorLaunchMaxDepthExceeded:
        return "cudaErrorLaunchMaxDepthExceeded";

    case cudaErrorLaunchFileScopedTex:
        return "cudaErrorLaunchFileScopedTex";

    case cudaErrorLaunchFileScopedSurf:
        return "cudaErrorLaunchFileScopedSurf";

    case cudaErrorSyncDepthExceeded:
        return "cudaErrorSyncDepthExceeded";

    case cudaErrorLaunchPendingCountExceeded:
        return "cudaErrorLaunchPendingCountExceeded";

    case cudaErrorNotPermitted:
        return "cudaErrorNotPermitted";

    case cudaErrorNotSupported:
        return "cudaErrorNotSupported";


    case cudaErrorHardwareStackError:
        return "cudaErrorHardwareStackError";

    case cudaErrorIllegalInstruction:
        return "cudaErrorIllegalInstruction";

    case cudaErrorMisalignedAddress:
        return "cudaErrorMisalignedAddress";

    case cudaErrorInvalidAddressSpace:
        return "cudaErrorInvalidAddressSpace";

    case cudaErrorInvalidPc:
        return "cudaErrorInvalidPc";

    case cudaErrorIllegalAddress:
        return "cudaErrorIllegalAddress";


    case cudaErrorInvalidPtx:
        return "cudaErrorInvalidPtx";

    case cudaErrorInvalidGraphicsContext:
        return "cudaErrorInvalidGraphicsContext";

    case cudaErrorStartupFailure:
        return "cudaErrorStartupFailure";

    case cudaErrorApiFailureBase:
        return "cudaErrorApiFailureBase";
    }

    return "<unknown>";
}

using namespace std;
string address_seed_in;
string address_key_in;
string address_amp_out;

uint32_t vertical_len;
uint32_t horizontal_len;
uint32_t vertical_block;
uint32_t horizontal_block;
uint32_t desired_block;
uint32_t key_blocks;
uint32_t* recv_key;
uint32_t* toeplitz_seed;
uint32_t* key_start;
uint32_t* key_start_zero_pos;
uint32_t* key_rest;
uint32_t* key_rest_zero_pos;
uint8_t* Output;
uint8_t* testMemoryHost;




atomic<uint32_t> input_cache_read_pos;
atomic<uint32_t> input_cache_write_pos;
atomic<uint32_t> output_cache_read_pos;
atomic<uint32_t> output_cache_write_pos;
mutex printlock;
float normalisation_float;
atomic<bool> unitTestsFailed = {false};
atomic<bool> unitTestBinInt2floatVerifyResultThreadFailed = {false};
atomic<bool> unitTestToBinaryArrayVerifyResultThreadFailed = {false};
atomic<bool> cuFFT_planned = {false};

 Complex c0_dev;
 Real h0_dev;
 Real h1_reduced_dev;
 Real normalisation_float_dev;
 uint32_t sample_size_dev;
 uint32_t pre_mul_reduction_dev;

 uint32_t intTobinMask_dev[32] =
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


 uint32_t ToBinaryBitShiftArray_dev[32] =
{

 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24



};


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

 void cudaAssertValue(uint32_t* data, uint32_t value) {
 uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
 
(static_cast <bool> (
data[i] == value
) ? void (0) : __assert_fail (
"data[i] == value"
, "PrivacyAmplification.cu", 189, __extension__ __PRETTY_FUNCTION__))
                        ;
}


int unitTestCalculateCorrectionFloat() {
 printlnStream(std::ostringstream().flush() << "Started CalculateCorrectionFloat Unit Test...");;
 bool unitTestsFailedLocal = false;
 cudaStream_t CalculateCorrectionFloatTestStream;
 cudaStreamCreate(&CalculateCorrectionFloatTestStream);
 uint32_t sample_size_test = pow(2, 8);
 uint32_t* count_one_of_global_seed_test;
 uint32_t* count_one_of_global_key_test;
 float* correction_float_dev_test;
 cudaMallocHost((void**)&count_one_of_global_seed_test, (134217728 / 1024) * sizeof(uint32_t));
 cudaMallocHost((void**)&count_one_of_global_key_test, (134217728 / 1024) * sizeof(uint32_t));
 cudaMallocHost((void**)&correction_float_dev_test, (134217728 / 1024) * sizeof(float));
 cudaMemcpyToSymbol(sample_size_dev, &sample_size_test, sizeof(uint32_t));
 for (uint32_t i = 0; i < sample_size_test; ++i) {
  for (uint32_t j = 0; j < sample_size_test; ++j) {
   *count_one_of_global_seed_test = i;
   *count_one_of_global_key_test = j;
   calculateCorrectionFloat
    (count_one_of_global_seed_test, count_one_of_global_key_test, correction_float_dev_test);
   cudaStreamSynchronize(CalculateCorrectionFloatTestStream);
   uint64_t cpu_count_multiplied = *count_one_of_global_seed_test * *count_one_of_global_key_test;
   double cpu_count_multiplied_normalized = cpu_count_multiplied / (double)sample_size_test;
   double count_multiplied_normalized_modulo = fmod(cpu_count_multiplied_normalized, 2.0);
   if (abs(*correction_float_dev_test - count_multiplied_normalized_modulo) > 0.0001) { std::cerr << "AssertionError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 216 << " on test case " << i * sample_size_test + j << ": Expected abs(" << *correction_float_dev_test - count_multiplied_normalized_modulo << ") < " << 0.0001 << endl; unitTestsFailed = true; unitTestsFailedLocal = true; };
  }
 }
 cudaMemcpyToSymbol(sample_size_dev, &sample_size, sizeof(uint32_t));
 printlnStream(std::ostringstream().flush() << "Completed CalculateCorrectionFloat Unit Test");;
 return unitTestsFailedLocal ? 100 : 0;
}


void calculateCorrectionFloat(uint32_t* count_one_of_global_seed, uint32_t* count_one_of_global_key, float* correction_float_dev)
{
 uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
 uint64_t count_multiplied = *(count_one_of_global_seed+i) * *(count_one_of_global_key+i);
 double count_multiplied_normalized = count_multiplied / (double)sample_size_dev;
 double two = 2.0;
 Real count_multiplied_normalized_modulo = (float)fmod(count_multiplied_normalized, two);
 *(correction_float_dev+i) = count_multiplied_normalized_modulo;
}


int unitTestSetFirstElementToZero() {
 printlnStream(std::ostringstream().flush() << "Started SetFirstElementToZero Unit Test...");;
 bool unitTestsFailedLocal = false;
 cudaStream_t SetFirstElementToZeroStreamTest;
 cudaStreamCreate(&SetFirstElementToZeroStreamTest);
 float* do1_test;
 float* do2_test;
 cudaMallocHost((void**)&do1_test, pow(2, 10) * 2 * sizeof(float));
 cudaMallocHost((void**)&do2_test, pow(2, 10) * 2 * sizeof(float));
 for (int i = 0; i < pow(2, 10) * 2; ++i) {
  do1_test[i] = i + 0.77;
  do2_test[i] = i + 0.88;
 }
 setFirstElementToZero
  (reinterpret_cast<Complex*>(do1_test), reinterpret_cast<Complex*>(do2_test));
 cudaStreamSynchronize(SetFirstElementToZeroStreamTest);
 if (abs(do1_test[0]) > 0.00001) { std::cerr << "AssertionError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 252 << " on test case " << 0 << ": Expected abs(" << do1_test[0] << ") < " << 0.00001 << endl; unitTestsFailed = true; unitTestsFailedLocal = true; };
 if (abs(do1_test[1]) > 0.00001) { std::cerr << "AssertionError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 253 << " on test case " << 1 << ": Expected abs(" << do1_test[1] << ") < " << 0.00001 << endl; unitTestsFailed = true; unitTestsFailedLocal = true; };
 if (abs(do2_test[0]) > 0.00001) { std::cerr << "AssertionError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 254 << " on test case " << 2 << ": Expected abs(" << do2_test[0] << ") < " << 0.00001 << endl; unitTestsFailed = true; unitTestsFailedLocal = true; };
 if (abs(do2_test[1]) > 0.00001) { std::cerr << "AssertionError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 255 << " on test case " << 3 << ": Expected abs(" << do2_test[1] << ") < " << 0.00001 << endl; unitTestsFailed = true; unitTestsFailedLocal = true; };
 for (int i = 2; i < pow(2, 10) * 2; ++i) {
  if (abs(do1_test[i] - (i + 0.77)) > 0.0001) { std::cerr << "AssertionError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 257 << " on test case " << i * 2 << ": Expected abs(" << do1_test[i] - (i + 0.77) << ") < " << 0.0001 << endl; unitTestsFailed = true; unitTestsFailedLocal = true; };
  if (abs(do2_test[i] - (i + 0.88)) > 0.0001) { std::cerr << "AssertionError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 258 << " on test case " << i * 2 + 1 << ": Expected abs(" << do2_test[i] - (i + 0.88) << ") < " << 0.0001 << endl; unitTestsFailed = true; unitTestsFailedLocal = true; };
 }
 printlnStream(std::ostringstream().flush() << "Completed SetFirstElementToZero Unit Test");;
 return unitTestsFailedLocal ? 100 : 0;
}


void setFirstElementToZero(Complex* do1, Complex* do2)
{
 uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
 if (i % 2 == 0) {
  do1[i / 2] = c0_dev;
 }
 else
 {
  do2[i / 2] = c0_dev;
 }
}


int unitTestElementWiseProduct() {
 printlnStream(std::ostringstream().flush() << "Started ElementWiseProduct Unit Test...");;
 bool unitTestsFailedLocal = false;
 cudaStream_t ElementWiseProductStreamTest;
 cudaStreamCreate(&ElementWiseProductStreamTest);
 uint32_t r = pow(2, 5);
 float* do1_test;
 float* do2_test;
 cudaMemcpyToSymbol(pre_mul_reduction_dev, &r, sizeof(uint32_t));
 cudaMallocHost((void**)&do1_test, pow(2, 10) * 2 * sizeof(float));
 cudaMallocHost((void**)&do2_test, pow(2, 10) * 2 * sizeof(float));
 for (int i = 0; i < pow(2, 10) * 2; ++i) {
  do1_test[i] = i + 0.77;
  do2_test[i] = i + 0.88;
 }
 ElementWiseProduct

  (reinterpret_cast<Complex*>(do1_test), reinterpret_cast<Complex*>(do2_test));
 cudaStreamSynchronize(ElementWiseProductStreamTest);
 for (int i = 0; i < pow(2, 10) * 2; i+=2) {
  float real = ((i + 0.77) / r) * ((i + 0.88) / r) - (((i + 1) + 0.77) / r) * (((i + 1) + 0.88) / r);
  float imag = ((i + 0.77) / r) * (((i + 1) + 0.88) / r) + (((i + 1) + 0.77) / r) * ((i + 0.88) / r);
  if (abs(do1_test[i] - real) > 0.001) { std::cerr << "AssertionError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 300 << " on test case " << i << ": Expected abs(" << do1_test[i] - real << ") < " << 0.001 << endl; unitTestsFailed = true; unitTestsFailedLocal = true; };
  if (abs(do1_test[i + 1] - imag) > 0.001) { std::cerr << "AssertionError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 301 << " on test case " << i + 1 << ": Expected abs(" << do1_test[i + 1] - imag << ") < " << 0.001 << endl; unitTestsFailed = true; unitTestsFailedLocal = true; };
 }
 cudaMemcpyToSymbol(pre_mul_reduction_dev, &pre_mul_reduction, sizeof(uint32_t));
 printlnStream(std::ostringstream().flush() << "Completed ElementWiseProduct Unit Test");;
 return unitTestsFailedLocal ? 100 : 0;
}



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
 register const Real float0 = 0.0f;
 register const Real float1_reduced = 1.0f / reduction;
 for (; i < i_max; ++i) {
  if (((i / 32) & (1 << (31 - (i % 32)))) == 0) {
   if (floatOutTest[i] != float0) { std::cerr << "AssertEqualsError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 338 << " on test case " << i << ": Expected " << float0 << " but it was " << floatOutTest[i] << endl; unitTestsFailed = true; unitTestsFailedLocal = true; }
  }
  else
  {
   if (floatOutTest[i] != float1_reduced) { std::cerr << "AssertEqualsError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 342 << " on test case " << i << ": Expected " << float1_reduced << " but it was " << floatOutTest[i] << endl; unitTestsFailed = true; unitTestsFailedLocal = true; }
  }
 }
 if (unitTestsFailedLocal) {
  unitTestBinInt2floatVerifyResultThreadFailed = true;
 }
}

int unitTestBinInt2float() {
 printlnStream(std::ostringstream().flush() << "Started TestBinInt2float Unit Test...");;
 atomic<bool> unitTestsFailedLocal = {false};
 cudaStream_t BinInt2floatStreamTest;
 cudaStreamCreate(&BinInt2floatStreamTest);
 uint32_t* binInTest;
 float* floatOutTest;
 cudaMallocHost((void**)&binInTest, (pow(2, 27) / 32) * sizeof(uint32_t));
 cudaMallocHost((void**)&floatOutTest, pow(2, 27) * sizeof(float));
 uint32_t* count_one_test;
 cudaMallocHost(&count_one_test, sizeof(uint32_t));

 const auto processor_count = std::thread::hardware_concurrency();
 for (int i = 0; i < pow(2, 27) / 32; ++i) {
  binInTest[i] = i;
 }
 unitTestBinInt2floatVerifyResultThreadFailed = false;
 for (uint32_t sample_size_test_exponent = 10; sample_size_test_exponent <= 27; ++sample_size_test_exponent)
 {
  int elementsToCheck = pow(2, sample_size_test_exponent);
  printlnStream(std::ostringstream().flush() << "TestBinInt2float Unit Test with 2^" << sample_size_test_exponent << " samples...");;
  uint32_t sample_size_test = elementsToCheck;
  uint32_t count_one_expected = A000788((sample_size_test/32)-1);
  *count_one_test = 0;
  memset(floatOutTest, 0xFF, pow(2, 27) * sizeof(float));
  binInt2float
                           (binInTest, floatOutTest, count_one_test);
  cudaStreamSynchronize(BinInt2floatStreamTest);
  if (*count_one_test != count_one_expected) { std::cerr << "AssertEqualsError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 378 << " on test case " << -1 << ": Expected " << count_one_expected << " but it was " << *count_one_test << endl; unitTestsFailed = true; unitTestsFailedLocal = true; };
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
 printlnStream(std::ostringstream().flush() << "Completed TestBinInt2float Unit Test");;
 return unitTestsFailedLocal ? 100 : 0;
}


void binInt2float(uint32_t* binIn, Real* realOut, uint32_t* count_one_global)
{

 Real h0_local = h0_dev;
 Real h1_reduced_local = h1_reduced_dev;
 uint32_t binInShared[32];

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
  atomicAdd(count_one_global + idx/sample_size_dev, 1);
  realOut[outPos] = h1_reduced_local;
 }
}

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

  data = ((((data) & 0xff000000) >> 24) |
    (((data) & 0x00ff0000) >> 8) |
    (((data) & 0x0000ff00) << 8) |
    (((data) & 0x000000ff) << 24));



  key_rest_little = key_rest_test[i / 32];
  key_rest_xor = ((((key_rest_little) & 0xff000000) >> 24) |
      (((key_rest_little) & 0x00ff0000) >> 8) |
      (((key_rest_little) & 0x0000ff00) << 8) |
      (((key_rest_little) & 0x000000ff) << 24));




  actualBit = (data & mask) > 0;
  expectedBit = ((i / 32) & mask) > 0;
  xorBit = (key_rest_xor & mask) > 0;

  expectedBit ^= xorBit;

  if (actualBit != expectedBit) { std::cerr << "AssertEqualsError in function " << __func__ << " in " << (strrchr("PrivacyAmplification.cu", '/') ? strrchr("PrivacyAmplification.cu", '/') + 1 : "PrivacyAmplification.cu") << ":" << 461 << " on test case " << i << ": Expected " << expectedBit << " but it was " << actualBit << endl; unitTestsFailed = true; unitTestsFailedLocal = true; }
 }
 if (unitTestsFailedLocal) {
  unitTestToBinaryArrayVerifyResultThreadFailed = true;
 }
}

int unitTestToBinaryArray() {
 printlnStream(std::ostringstream().flush() << "Started ToBinaryArray Unit Test...");;
 atomic<bool> unitTestsFailedLocal = {false};
 cudaStream_t ToBinaryArrayStreamTest;
 cudaStreamCreate(&ToBinaryArrayStreamTest);
 register const Real float0 = 0.0f;
 register const Real float1 = 1.0f;
 float* invOutTest;
 uint32_t* binOutTest;
 uint32_t* key_rest_test;
 Real* correction_float_dev_test;
 cudaMallocHost((void**)&invOutTest, 134217728 * sizeof(float));
 cudaMallocHost((void**)&binOutTest, 16777216 * sizeof(uint32_t));
 cudaMallocHost((void**)&key_rest_test, 16777216 * sizeof(uint32_t));
 cudaMallocHost((void**)&correction_float_dev_test, (134217728 / 1024) * sizeof(Real));
 memset(key_rest_test, 0b10101010, 16777216 * sizeof(uint32_t));
 *correction_float_dev_test = 1.9f;
 uint32_t normalisation_float_test = 1.0f;
 cudaMemcpyToSymbol(normalisation_float_dev, &normalisation_float_test, sizeof(uint32_t));
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
  printlnStream(std::ostringstream().flush() << "ToBinaryArray Unit Test with 2^" << sample_size_test_exponent << " samples...");;
  memset(binOutTest, 0xCC, (pow(2, 27) / 32) * sizeof(uint32_t));
  ToBinaryArray (invOutTest, binOutTest, key_rest_test, correction_float_dev_test, (int)((int)(vertical_block_test) / 31) + 1);
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
 cudaMemcpyToSymbol(normalisation_float_dev, &normalisation_float, sizeof(uint32_t));
 printlnStream(std::ostringstream().flush() << "Completed ToBinaryArray Unit Test");;
 return unitTestsFailedLocal ? 100 : 0;
}


void ToBinaryArray(Real* invOut, uint32_t* binOut, uint32_t* key_rest_local, Real* correction_float_dev, uint32_t block_index_divisor)
{
 const Real normalisation_float_local = normalisation_float_dev;
 const uint32_t block = blockIdx.x;
 const uint32_t idx = threadIdx.x;
 const Real correction_float = *correction_float_dev+block/block_index_divisor;

 uint32_t key_rest_xor[31];
 uint32_t binOutRawBit[992];
 if (idx < 992) {
  binOutRawBit[idx] = ((__float2int_rn(invOut[block * 992 + idx] / normalisation_float_local + correction_float) & 1)
   << ToBinaryBitShiftArray_dev[idx % 32]);
 }
 else if (idx < 1023)
 {


  uint32_t key_rest_little = key_rest_local[block * 31 + idx - 992];
  key_rest_xor[idx - 992] =
   ((((key_rest_little) & 0xff000000) >> 24) |
    (((key_rest_little) & 0x00ff0000) >> 8) |
    (((key_rest_little) & 0x0000ff00) << 8) |
    (((key_rest_little) & 0x000000ff) << 24));




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

    ^ key_rest_xor[idx]

    ;
  binOut[block * 31 + idx] = binOutLocal;
 }
}


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
 printlnStream(std::ostringstream().flush() << std::fixed << std::setprecision(8) << result.first << " | " << result.second);;
 return abs(result.first - expectedSum1) < allowedAbsDeltaSum1 && abs(result.second - expectedSum2) < allowedAbsDeltaSum2;
}

inline void key2StartRest() {
 uint32_t* key_start_block = key_start + 8388608 * input_cache_write_pos;
 uint32_t* key_rest_block = key_rest + 8388608 * input_cache_write_pos;
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

 if (seedfile_length < 16777216)
 {
  cout << "File \"" << toeplitz_seed_path << "\" is with " << seedfile_length << " bytes too short!" << endl;
  cout << "it is required to be at least " << 16777216 << " bytes => terminating!" << endl;
  exit(104);
  abort();
 }

 char* toeplitz_seed_char = reinterpret_cast<char*>(toeplitz_seed + 16777216 * input_cache_write_pos);
 seedfile.read(toeplitz_seed_char, 16777216);
 for (uint32_t i = 0; i < input_banks_to_cache; ++i) {
  uint32_t* toeplitz_seed_block = toeplitz_seed + 16777216 * i;
  memcpy(toeplitz_seed_block, toeplitz_seed, 16777216);
 }
}


inline void readKeyFromFile() {



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

 if (keyfile_length < key_blocks * sizeof(uint32_t) * blocks_in_bank)
 {
  cout << "File \"" << keyfile_path << "\" is with " << keyfile_length << " bytes too short!" << endl;
  cout << "it is required to be at least " << key_blocks * sizeof(uint32_t) * blocks_in_bank << " bytes => terminating!" << endl;
  exit(106);
  abort();
 }

 char* recv_key_char = reinterpret_cast<char*>(recv_key);
 keyfile.read(recv_key_char, key_blocks * sizeof(uint32_t) * blocks_in_bank);
 key2StartRest();
 for (uint32_t i = 0; i < input_banks_to_cache; ++i) {
  uint32_t* key_start_block = key_start + desired_block * i;
  uint32_t* key_rest_block = key_rest + desired_block * i;
  uint32_t* key_start_zero_pos_block = key_start_zero_pos + i;
  uint32_t* key_rest_zero_pos_block = key_rest_zero_pos + i;
  memcpy(key_start_block, key_start, desired_block * sizeof(uint32_t));
  memcpy(key_rest_block, key_rest, desired_block * sizeof(uint32_t));
  *key_start_zero_pos_block = *key_start_zero_pos;
  *key_rest_zero_pos_block = *key_rest_zero_pos;
 }
}


void reciveData() {
 return;
 void* socket_seed_in = nullptr;
 void* socket_key_in = nullptr;
 void* context_seed_in = nullptr;
 void* context_key_in = nullptr;
 int timeout_seed_in = 1000;
 int timeout_key_in = 1000;

 if (use_matrix_seed_server)
 {
  context_seed_in = zmq_ctx_new();
  socket_seed_in = zmq_socket(context_seed_in, 
                                              3
                                                     );
  zmq_setsockopt(socket_seed_in, 
                                27
                                            , &timeout_seed_in, sizeof(int));
  zmq_connect(socket_seed_in, address_seed_in.c_str());
 }
 else
 {
  readMatrixSeedFromFile();
 }

 if (use_key_server)
 {
  context_key_in = zmq_ctx_new();
  socket_key_in = zmq_socket(context_key_in, 
                                            3
                                                   );
  zmq_setsockopt(socket_key_in, 
                               27
                                           , &timeout_key_in, sizeof(int));
  zmq_connect(socket_key_in, address_key_in.c_str());
 }
 else
 {
  readKeyFromFile();
 }

 bool recive_toeplitz_matrix_seed = use_matrix_seed_server;
 while (true)
 {

  while (input_cache_write_pos % input_banks_to_cache == input_cache_read_pos) {
   this_thread::yield();
  }

  uint32_t* toeplitz_seed_block = toeplitz_seed + 8388608 * input_cache_write_pos;
  if (recive_toeplitz_matrix_seed) {
  retry_receiving_seed:
   zmq_send(socket_seed_in, "SYN", 3, 0);
   if (zmq_recv(socket_seed_in, toeplitz_seed_block, 16777216, 0) != 16777216) {
    printlnStream(std::ostringstream().flush() << "Error receiving data from Seedserver! Retrying...");;
    zmq_close(context_seed_in);
    socket_seed_in = zmq_socket(context_seed_in, 
                                                3
                                                       );
    zmq_setsockopt(socket_seed_in, 
                                  27
                                              , &timeout_seed_in, sizeof(int));
    zmq_connect(socket_seed_in, address_seed_in.c_str());
    goto retry_receiving_seed;
   }
   if (show_zeromq_status) {
    printlnStream(std::ostringstream().flush() << "Seed Block recived");;
   }

   if (!dynamic_toeplitz_matrix_seed)
   {
    recive_toeplitz_matrix_seed = false;
    zmq_disconnect(socket_seed_in, address_seed_in.c_str());
    zmq_close(socket_seed_in);
    zmq_ctx_destroy(socket_seed_in);
    for (uint32_t i = 0; i < input_banks_to_cache; ++i) {
     uint32_t* toeplitz_seed_block = toeplitz_seed + 8388608 * i;
     memcpy(toeplitz_seed_block, toeplitz_seed, 8388608);
    }
   }
  }

  if (use_key_server)
  {
  retry_receiving_key:
   if (zmq_send(socket_key_in, "SYN", 3, 0) != 3) {
    printlnStream(std::ostringstream().flush() << "Error sending SYN to Keyserver! Retrying...");;
    goto retry_receiving_key;
   }
   if (zmq_recv(socket_key_in, &vertical_block, sizeof(uint32_t), 0) != sizeof(uint32_t)) {
    printlnStream(std::ostringstream().flush() << "Error receiving vertical_blocks from Keyserver! Retrying...");;
    zmq_close(context_key_in);
    socket_key_in = zmq_socket(context_key_in, 
                                              3
                                                     );
    zmq_setsockopt(socket_key_in, 
                                 27
                                             , &timeout_key_in, sizeof(int));
    zmq_connect(socket_key_in, address_key_in.c_str());
    goto retry_receiving_key;
   }
   vertical_len = vertical_block * 32;
   horizontal_len = sample_size - vertical_len;
   horizontal_block = horizontal_len / 32;
   if (zmq_recv(socket_key_in, recv_key, key_blocks * sizeof(uint32_t), 0) != key_blocks * sizeof(uint32_t)) {
    printlnStream(std::ostringstream().flush() << "Error receiving data from Keyserver! Retrying...");;
    zmq_close(context_key_in);
    socket_key_in = zmq_socket(context_key_in, 
                                              3
                                                     );
    zmq_setsockopt(socket_key_in, 
                                 27
                                             , &timeout_key_in, sizeof(int));
    zmq_connect(socket_key_in, address_key_in.c_str());
    goto retry_receiving_key;
   }
   if (show_zeromq_status) {
    printlnStream(std::ostringstream().flush() << "Key Block recived");;
   }
   key2StartRest();
  }

  input_cache_write_pos = (input_cache_write_pos + 1) % input_banks_to_cache;
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
  ss << std::uppercase << std::hex << "0x" << std::setw(2) << std::setfill('0') << (int)data[i];
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

 return memcmp(calculatedHash, expectedHash, 32) == 0;
}

void verifyData(const uint8_t* dataToVerify) {
 if (isSha3(dataToVerify, (vertical_len / 8) * blocks_in_bank, ampout_sha3)) {
  printlnStream(std::ostringstream().flush() << "VERIFIED!");;
 }
 else
 {
  printlnStream(std::ostringstream().flush() << "VERIFICATION FAILED!");;

 }
}


void sendData() {
 int32_t rc;
 char syn[3];
 void* amp_out_socket = nullptr;
 if (host_ampout_server)
 {
  void* amp_out_context = zmq_ctx_new();
  amp_out_socket = zmq_socket(amp_out_context, 
                                              4
                                                     );
  while (zmq_bind(amp_out_socket, address_amp_out.c_str()) != 0) {
   printlnStream(std::ostringstream().flush() << "Binding to \"" << address_amp_out << "\" failed! Retrying...");;
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

  while ((output_cache_read_pos + 1) % output_banks_to_cache == output_cache_write_pos) {
   this_thread::yield();
  }
  output_cache_read_pos = (output_cache_read_pos + 1) % output_banks_to_cache;

  uint8_t* output_block = Output + 16777216 * output_cache_read_pos;

  if (verify_ampout)
  {
   verifyDataPool->enqueue(verifyData, output_block);
  }

  if (ampOutsToStore != 0) {
   if (ampOutsToStore > 0) {
    --ampOutsToStore;
   }
   ampout_file.write((char*)&output_block[0], (vertical_len / 8) * blocks_in_bank);
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
    printlnStream(std::ostringstream().flush() << "Error receiving SYN! Retrying...");;
    goto retry_sending_amp_out;
   }
   if (zmq_send(amp_out_socket, output_block, vertical_len / 8, 0) != vertical_len / 8) {
    printlnStream(std::ostringstream().flush() << "Error sending data to AMPOUT client! Retrying...");;
    goto retry_sending_amp_out;
   }
   if (show_zeromq_status) {
    printlnStream(std::ostringstream().flush() << "Block sent to AMPOUT Client");;
   }
  }

  stop = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(stop - start).count();
  start = chrono::high_resolution_clock::now();

  if (show_ampout >= 0)
  {
   printlock.lock();
   cout << "Blocktime: " << duration / 1000.0 << " ms => " << (1000000.0 / duration) * (134217728 / 1000000.0) << " Mbit/s" << endl;
   if (show_ampout > 0)
   {
    for (size_t i = 0; i < (((vertical_block * sizeof(uint32_t)) < (show_ampout)) ? (vertical_block * sizeof(uint32_t)) : (show_ampout)); ++i)
    {
     printf("q0x%02X: %s\n", output_block[i], bitset<8>(output_block[i]).to_string().c_str());
    }
   }
   fflush(
         stdout
               );
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


 address_seed_in = root["address_seed_in"].As<string>("tcp://127.0.0.1:45555");
 address_key_in = root["address_key_in"].As<string>("tcp://127.0.0.1:47777");
 address_amp_out = root["address_amp_out"].As<string>("tcp://127.0.0.1:48888");

 sample_size = static_cast<int>(round(pow(2, root["factor_exp"].As<uint32_t>(27))));
 reduction = static_cast<int>(round(pow(2, root["reduction_exp"].As<uint32_t>(11))));
 pre_mul_reduction = static_cast<int>(round(pow(2, root["pre_mul_reduction_exp"].As<uint32_t>(5))));
 cuda_device_id_to_use = root["cuda_device_id_to_use"].As<uint32_t>(1);
 input_banks_to_cache = root["input_blocks_to_cache"].As<uint32_t>(16);
 output_banks_to_cache = root["output_blocks_to_cache"].As<uint32_t>(16);

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

 blocks_in_bank = 134217728 / sample_size;
 vertical_len = sample_size / 4 + sample_size / 8;
 horizontal_len = sample_size / 2 + sample_size / 8;
 vertical_block = vertical_len / 32;
 horizontal_block = horizontal_len / 32;
 desired_block = sample_size / 32;
 key_blocks = desired_block + 1;
 recv_key = (uint32_t*)malloc(17301504);
 key_start_zero_pos = (uint32_t*)malloc(input_banks_to_cache * sizeof(uint32_t));
 key_rest_zero_pos = (uint32_t*)malloc(input_banks_to_cache * sizeof(uint32_t));
}


inline void setConsoleDesign(){}
int main(int argc, char* argv[])
{

 string border(about.length(), '#');
 cout << border << endl << about << endl << border << endl << endl;

 readConfig();

 cout << "#PrivacyAmplification with " << sample_size << " bits" << endl << endl;
 cudaSetDevice(cuda_device_id_to_use);
 setConsoleDesign();

 input_cache_read_pos = input_banks_to_cache - 1;
 input_cache_write_pos = 0;
 output_cache_read_pos = input_banks_to_cache - 1;
 output_cache_write_pos = 0;

 uint32_t* count_one_of_global_seed;
 uint32_t* count_one_of_global_key;
 float* correction_float_dev;
 Real* di1;
 Real* di2;
 Real* invOut;
 Complex* do1;
 Complex* do2;
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


 cudaEvent_t start;
 cudaEvent_t stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);


 cudaMallocHost((void**)&toeplitz_seed, 16777216 * input_banks_to_cache);
 cudaMallocHost((void**)&key_start, 16777216 * input_banks_to_cache);
 cudaMallocHost((void**)&key_rest, 16777216 * (input_banks_to_cache+1));
 cudaMallocHost((void**)&Output, 16777216 * output_banks_to_cache + 992);
 fill(key_start_zero_pos, key_start_zero_pos + input_banks_to_cache, desired_block);
 fill(key_rest_zero_pos, key_rest_zero_pos + input_banks_to_cache, desired_block);


 cudaMalloc(&count_one_of_global_seed, (134217728 / 1024) * sizeof(uint32_t));
 cudaMalloc(&count_one_of_global_key, (134217728 / 1024) * sizeof(uint32_t));
 cudaMalloc(&correction_float_dev, (134217728 / 1024) * sizeof(float));
 cudaMalloc((void**)&di1, 134217728 * sizeof(Real)); cudaMemset(*(void**)&di1, 0b00000000, 134217728 * sizeof(Real));;



 cudaMalloc((void**)&di2, (134217728 + 992) * sizeof(Real));


 cudaMalloc((void**)&do1, 134217728 * sizeof(Complex));



 cudaMalloc((void**)&do2, max(134217728 * sizeof(Complex), (134217728 + 992) * sizeof(Real)));

 register const Complex complex0 = make_float2(0.0f, 0.0f);
 register const Real float0 = 0.0f;
 register const Real float1_reduced = 1.0f / reduction;
 const uint32_t total_reduction = reduction * pre_mul_reduction;
 normalisation_float = ((float)sample_size) / ((float)total_reduction) / ((float)total_reduction);


 cudaMemcpyToSymbol(c0_dev, &complex0, sizeof(Complex));
 cudaMemcpyToSymbol(h0_dev, &float0, sizeof(float));
 cudaMemcpyToSymbol(h1_reduced_dev, &float1_reduced, sizeof(float));
 cudaMemcpyToSymbol(normalisation_float_dev, &normalisation_float, sizeof(float));
 cudaMemcpyToSymbol(sample_size_dev, &sample_size, sizeof(uint32_t));
 cudaMemcpyToSymbol(pre_mul_reduction_dev, &pre_mul_reduction, sizeof(uint32_t));


 thread threadReciveObj(reciveData);
 threadReciveObj.detach();


 thread threadSendObj(sendData);
 threadSendObj.detach();


 cufftHandle plan_forward_R2C;
 cufftHandle plan_inverse_C2R;
 if (cuFFT_planned) { cufftDestroy(plan_forward_R2C); cufftDestroy(plan_inverse_C2R); } cufftResult result_forward_FFT = cufftPlan1d(&plan_forward_R2C, sample_size, CUFFT_R2C, blocks_in_bank); if (result_forward_FFT != CUFFT_SUCCESS) { printlnStream(std::ostringstream().flush() << "Failed to plan FFT 1! Error Code: " << result_forward_FFT);; exit(0); } cufftResult result_inverse_FFT = cufftPlan1d(&plan_inverse_C2R, sample_size, CUFFT_C2R, blocks_in_bank); if (result_inverse_FFT != CUFFT_SUCCESS) { printlnStream(std::ostringstream().flush() << "Failed to plan IFFT 1! Error Code: " << result_inverse_FFT);; exit(0); } cuFFT_planned = true;;


 uint32_t relevant_keyBlocks = horizontal_block + 1;
 uint32_t relevant_keyBlocks_old = 0;

 bool recalculate_toeplitz_matrix_seed = true;
 bool speedtest = false;
 bool doTest = true;
 uint32_t dist_freq = sample_size / 2 + 1;
 invOut = reinterpret_cast<Real*>(do2);

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
     blocks_in_bank = 134217728 / sample_size;
     vertical_len = sample_size / 4 + sample_size / 8;
     horizontal_len = sample_size / 2 + sample_size / 8;
     vertical_block = vertical_len / 32;
     horizontal_block = horizontal_len / 32;
     desired_block = sample_size / 32;
     key_blocks = desired_block + 1;
     relevant_keyBlocks = horizontal_block + 1;
     normalisation_float = ((float)sample_size) / ((float)total_reduction) / ((float)total_reduction);
     cudaMemcpyToSymbol(normalisation_float_dev, &normalisation_float, sizeof(float));
     cudaMemcpyToSymbol(sample_size_dev, &sample_size, sizeof(uint32_t));
     dist_freq = sample_size / 2 + 1;
     if (cuFFT_planned) { cufftDestroy(plan_forward_R2C); cufftDestroy(plan_inverse_C2R); } cufftResult result_forward_FFT = cufftPlan1d(&plan_forward_R2C, sample_size, CUFFT_R2C, blocks_in_bank); if (result_forward_FFT != CUFFT_SUCCESS) { printlnStream(std::ostringstream().flush() << "Failed to plan FFT 1! Error Code: " << result_forward_FFT);; exit(0); } cufftResult result_inverse_FFT = cufftPlan1d(&plan_inverse_C2R, sample_size, CUFFT_C2R, blocks_in_bank); if (result_inverse_FFT != CUFFT_SUCCESS) { printlnStream(std::ostringstream().flush() << "Failed to plan IFFT 1! Error Code: " << result_inverse_FFT);; exit(0); } cuFFT_planned = true;
     for (int k = 0; k < 10; ++k) {

      goto mainloop;
      return_speedtest:;

      stop = chrono::high_resolution_clock::now();
      auto duration = chrono::duration_cast<chrono::microseconds>(stop - start).count();
      start = chrono::high_resolution_clock::now();
      printlnStream(std::ostringstream().flush() << "d[" << i << "," << j << "," << k << "]=" << (1000000.0 / duration) * (134217728 / 1000000.0));;
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





 while (true) {


  while ((input_cache_read_pos + 1) % input_banks_to_cache == input_cache_write_pos) {
   this_thread::yield();
  }
  input_cache_read_pos = (input_cache_read_pos + 1) % input_banks_to_cache;
  mainloop:
  input_cache_read_pos = 0;


  relevant_keyBlocks_old = relevant_keyBlocks;
  relevant_keyBlocks = horizontal_block + 1;
  if (relevant_keyBlocks_old > relevant_keyBlocks) {

   uint32_t amout_of_zeros_to_fill = (relevant_keyBlocks_old - relevant_keyBlocks);
   uint32_t key_block_size = relevant_keyBlocks + amout_of_zeros_to_fill;
   for (int i = 0; i < blocks_in_bank; ++i) {
    cudaMemset(di1 + i*key_block_size + relevant_keyBlocks, 0b00000000, amout_of_zeros_to_fill * sizeof(Real));
   }
  }

  cudaMemset(count_one_of_global_key, 0b00000000, blocks_in_bank * sizeof(uint32_t));







  binInt2float
                          (key_start + 8388608 * input_cache_read_pos, di1, count_one_of_global_key);
  if (recalculate_toeplitz_matrix_seed) {
   cudaMemset(count_one_of_global_seed, 0b00000000, blocks_in_bank * sizeof(uint32_t));






   binInt2float
                            (toeplitz_seed + 8388608 * input_cache_read_pos, di2, count_one_of_global_seed);
   cudaStreamSynchronize(BinInt2floatSeedStream);
  }
  cudaStreamSynchronize(BinInt2floatKeyStream);






  calculateCorrectionFloat
   (count_one_of_global_key, count_one_of_global_seed, correction_float_dev);






  cufftExecR2C(plan_forward_R2C, di1, do1);
  if (recalculate_toeplitz_matrix_seed) {






   cufftExecR2C(plan_forward_R2C, di2, do2);
   if (!dynamic_toeplitz_matrix_seed)
   {
    recalculate_toeplitz_matrix_seed = false;
    invOut = reinterpret_cast<Real*>(di2);
   }
  }
  cudaStreamSynchronize(FFTStream);
  cudaStreamSynchronize(CalculateCorrectionFloatStream);
  setFirstElementToZero (do1, do2);
  cudaStreamSynchronize(ElementWiseProductStream);
  ElementWiseProduct (do1, do2);
  cudaStreamSynchronize(ElementWiseProductStream);
  cufftExecC2R(plan_inverse_C2R, do1, invOut);
  cudaStreamSynchronize(FFTStream);


  if (!speedtest) {
   while (output_cache_write_pos % output_banks_to_cache == output_cache_read_pos) {
    this_thread::yield();
   }
  }


  uint32_t* binOut = reinterpret_cast<uint32_t*>(Output + 16777216 * output_cache_write_pos);
  ToBinaryArray
   (invOut, binOut, key_rest + 8388608 * input_cache_read_pos, correction_float_dev, (vertical_block / 31) + 1);
  cudaStreamSynchronize(ToBinaryArrayStream);
  if (speedtest) {

  }
  else
  {
   output_cache_write_pos = (output_cache_write_pos + 1) % output_banks_to_cache;
  }

 }



 cufftDestroy(plan_forward_R2C);
 cufftDestroy(plan_inverse_C2R);


 cudaFree(di1);
 cudaFree(di2);
 cudaFree(invOut);
 cudaFree(do1);
 cudaFree(do2);
 cudaFree(Output);


 cudaEventDestroy(start);
 cudaEventDestroy(stop);

 return 0;
}
