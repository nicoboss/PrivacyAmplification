#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

typedef float    Real;
typedef float2   Complex;

void printStream(std::ostream& os);

void printlnStream(std::ostream& os);

/// @return The sum of both parameters.
std::string convertStreamToString(std::ostream& os);

__global__ void calculateCorrectionFloat(uint32_t* count_one_global_seed, uint32_t* count_one_global_key, float* correction_float_dev);

__global__ void setFirstElementToZero(Complex* do1, Complex* do2);

__global__ void ElementWiseProduct(Complex* do1, Complex* do2);

__global__ void ToFloatArray(uint32_t n, uint32_t b, Real* floatOut, Real normalisation_float);

__global__ void ToBinaryArray(Real* invOut, uint32_t* binOut, uint32_t* key_rest_local, Real* correction_float_dev);

__global__ void binInt2float(uint32_t* binIn, Real* realOut, uint32_t* count_one_global);

void intToBinCPU(int* intIn, uint32_t* binOut, uint32_t outSize);

/// @brief Prints bynary byte data.
/// @param position The memory location where the data to print start
/// @param end The memory location where the data to print end
/// Loops from start to end and prints each bytes binary representation
/// After printing all binary data a newline character is printed
void printBin(const uint8_t* position, const uint8_t* end);

/// @brief Prints bynary integer data.
/// @param position The memory location where the data to print start
/// @param end The memory location where the data to print end
/// Loops from start to end and prints each integers binary representation
/// After printing all binary data a newline character is printed
void printBin(const uint32_t* position, const uint32_t* end);

void key2StartRest();

void reciveData();

void verifyData(const unsigned char* dataToVerify);

void sendData();

void readConfig();

int main(int argc, char* argv[]);
