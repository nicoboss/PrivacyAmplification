#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

//Real numbers are stored as single precision float
typedef float    Real;
//Complex numbers are stored as two single precision floats glued together
typedef float2   Complex;

#define TRUE 1
#define FALSE 0
#define VERSION "1.0"
//Return the minimum of two values
#define min_template(a,b) (((a) < (b)) ? (a) : (b))

/*In modified toeplitz hashing the result needs to be XORed with key_rest
  This can be disabled for debugging purposes but must be enabled for security!
  On purpose not in config.yaml for security and performance!*/
#define XOR_WITH_KEY_REST TRUE

/*By default the result will be in 4 byte little endian.
  If big endian is required, set the following definition to true otherwise to false.
  This has a very small perfromance impact.
  Due to performance reasons this can't be set in config.yaml*/
#define AMPOUT_REVERSE_ENDIAN TRUE

/*Enable debug output required for debugging purposes.
  On purpose not in config.yaml as this is only meant to be used by developers*/
#define SHOW_DEBUG_OUTPUT FALSE

/*Enable input debug output required for debugging purposes like markix seed and key.
  On purpose not in config.yaml as this is only meant to be used by developers*/
#define SHOW_INPUT_DEBUG_OUTPUT FALSE

/*Privacy Amplification input size in bits
  Has to be 2^x and 2^27 is the maximum*/
uint32_t sample_size;

/*FFT input maps binary 0, 1 to 0 and 1/reduction which
  will be corrected during normalisation after IFFT
  This has an impact on the Privacy Amplification precision*/
uint32_t reduction;

/*After the FFT before the element wise multiplication
  every element will be devided by pre_mul_reduction
  This has an impact on the Privacy Amplification precision*/
uint32_t pre_mul_reduction;

/*Specifies which GPU to use by setting this value to the CUDA device ID.
  Which ID matches to which GPU can be seen using nvidia-smi (on Linux and Windows)*/
uint32_t cuda_device_id_to_use;

/*Specifies how large the input cache should be. If Privacy Amplification is slower
  then the data input cache will fill up. Cache requires RAM.
  Its value must be 2 or larger while at 16 or higher is recommended.*/
uint32_t input_blocks_to_cache;

/*Specifies how large the ouput cache should be. If the data reciever is slower
  then the Privacy Amplification this cache will fill up. Cache requires RAM.
  Its value must be 2 or larger while at 16 or higher is recommended.*/
uint32_t output_blocks_to_cache;

/*Specifies if the toeplitz matrix seed should be exchanged for every Privacy Amplification
  This has a huge performance and security impact. Not changing it will make the algorithms
  security to be no longer  proofen to be secure while changing
  it every time will reduce performance by around 33% (around 47% faster).
  I highly recommend to leave this enabled if security matters.*/
bool dynamic_toeplitz_matrix_seed;

/*Displays Blocktime, Mbit/s input throughput and the first 4 bytes of the final
  Privacy Amplification result to the screen and only has a very little
  impact on performance. I would leave it on but feel free to disable that.*/
bool show_ampout;

/*If enabled connects to the matrix seed server on address_seed_in to request the toeplitz
  matrix seed for the current block. If dynamic_toeplitz_matrix_seed is disabled and this 
  enabled only one block at the program start will be requested. The matrix seed server
  ensures all parties envolved will receive the same seed for the same block.
  Warning: Currently the channel to the matrix seed server is not authenticated and has 
  to be implmented before any real world use. This is planned to be done.
  
  If disabled the matrix seed will be read from the path specified in toeplitz_seed_path
  however this only makes sense if dynamic_toeplitz_matrix_seed is disabled
  or for testcases as only one block worth of data will be ever read from that file and
  copied input_blocks_to_cache times to fill the input cache.*/
bool use_matrix_seed_server;
std::string toeplitz_seed_path;

/*If enabled connects to the key server on address_key_in to request the key for the
  current block.
  Warning: The server has to be on the same computer as the key gets transmitted insecurrely
  
  If disabled they key will be read from the path specified in keyfile_path however this
  only makes sense for testcases as only one block worth of data will be ever read from
  that file and copied input_blocks_to_cache to fill the input cache.*/
bool use_key_server;
std::string keyfile_path;

/*If enabled connects to the ampout client on address_amp_out to send the Privacy
  Amplification result.
  
  If disabled no output will be sent anywhere which only makes sense for debugging
  if either verify_ampout, store_first_ampouts_in_file, show_ampout is enabled or
  the preprocessor definition SHOW_DEBUG_OUTPUT is set.*/
bool host_ampout_server;

/*Stores the first n Privace amplification outputs to ampout.bin
  Set to 0 to display this and to -1 to store all output to ampout.bin.*/
int32_t store_first_ampouts_in_file;

/*If enabled verifies if the result of the Privacy Amplification of the provided
  keyfile.bin and toeplitz_seed.bin with a sample_size of 2^27 and a compression
  factor of vertical = sample_size / 4 + sample_size / 8 matches the SHA3-256
  hash of  C422B6865C72CAD82CC26A1462B8A4566F911750F31B1475691269C1B7D4A716.
  This result was verified with a python reference implementation and ensures
  during development that correctness of this Privacy Amplification implementation.
  Disable this if you are using anything else then the provided testdata with above
  settings. A verification error will cause the programm to exit with error 101.*/
bool verify_ampout;

/*Specifies how many threads for the ampout verification should be used. Around
  1.3 Gbit/s input throughput per thread with a sample_size of 2^27 and a
  compression factor of vertical = sample_size / 4 + sample_size / 8
  I recommend setting this value to 4 or higher to not bottleneck performance*/
uint32_t verify_ampout_threads;

#define print(TEXT) printStream(std::ostringstream().flush() << TEXT);
#define println(TEXT) printlnStream(std::ostringstream().flush() << TEXT);
#define streamToString(TEXT) convertStreamToString(std::ostringstream().flush() << TEXT);
#define cudaCalloc(a,b) if (cudaMalloc(a, b) == cudaSuccess) cudaMemset(*a, 0b00000000, b);

const uint8_t ampout_sha3[] = { 0xC4, 0x22, 0xB6, 0x86, 0x5C, 0x72, 0xCA, 0xD8,
                               0x2C, 0xC2, 0x6A, 0x14, 0x62, 0xB8, 0xA4, 0x56,
                               0x6F, 0x91, 0x17, 0x50, 0xF3, 0x1B, 0x14, 0x75,
                               0x69, 0x12, 0x69, 0xC1, 0xB7, 0xD4, 0xA7, 0x16 };

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
