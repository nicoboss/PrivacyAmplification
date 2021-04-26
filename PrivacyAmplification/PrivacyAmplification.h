#pragma once

#if defined(__NVCC__)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#else
struct float2
{
    float re, im;
};
//Complex numbers are stored as two single precision floats glued together
typedef float2 Complex;
#endif

//Real numbers are stored as single precision float
typedef float    Real;
//Complex numbers are stored as two single precision floats glued together
typedef float2   Complex;

#define TRUE 1
#define FALSE 0
#define VERSION "1.1.0"
#define STOPWATCH TRUE
#define NDEBUG

//Return the minimum of two values
#define min_template(a,b) (((a) < (b)) ? (a) : (b))

/*In modified toeplitz hashing the result needs to be XORed with key_rest
  This can be disabled for debugging purposes but must be enabled for security!
  On purpose not in config.yaml for security and performance!*/
#define XOR_WITH_KEY_REST TRUE

/*If big endian is required, set the following definition to true otherwise to false.
  Enabling this has a very small perfromance impact.
  Due to performance reasons this can't be set in config.yaml*/
#define AMPOUT_REVERSE_ENDIAN TRUE

/*Enable debug output required for debugging purposes.
  On purpose not in config.yaml as this is only meant to be used by developers*/
#define SHOW_DEBUG_OUTPUT FALSE

/*Enable input debug output required for debugging purposes like markix seed and key.
  On purpose not in config.yaml as this is only meant to be used by developers*/
#define SHOW_INPUT_DEBUG_OUTPUT FALSE

/*Privacy Amplification input size in bits
  Has to be 2^x and 2^27 is the maximum
  Needs to match with the one specified in other components*/
uint32_t sample_size;
uint64_t bufferSize;
uint64_t bufferSizeInput;
uint8_t* value_dev;
uint32_t zero_cpu = 0;

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

/*Displays the first n bytes of the final Privacy Amplification result
to the console.n > 0 has a little performance impact.
Set this to - 1 to also disable displaying Blocktime and input throughput in Mbit/s*/
int32_t show_ampout;

/*Displays ZeroMQ status messages which could have a very little performance impact.*/
bool show_zeromq_status;

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

/*If enabled connects to the key server on address_key_in to request the raw corrected key
  for the current block.
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
  hash of C422B6865C72CAD82CC26A1462B8A4566F911750F31B1475691269C1B7D4A716.
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

/*streams are awesome in C++ but we want to be able to call a funtion line
println("Hallo " << name) which requires the following function-like macros*/

/*Prints a stream in a thread save way.
  Example: print("Hallo " << name)*/
#define print(TEXT) printStream(std::ostringstream().flush() << TEXT);

/*Prints a stream in a thread save way and adds a newline at the end.
  Example: println("Hallo " << name)*/
#define println(TEXT) printlnStream(std::ostringstream().flush() << TEXT);

/*Convearts a stream into an std::string
  Example: std::string greeting = streamToString("Hallo " << name);*/
#define streamToString(TEXT) convertStreamToString(std::ostringstream().flush() << TEXT);

/*SHA3-256 Hash of the provided keyfile.bin and toeplitz_seed.bin with a sample_size of 2^27
and a compression factor of vertical = sample_size / 4 + sample_size / 8 which was verified
with a python reference implementation and is used for correctness testing*/
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

/// @brief Prints a stream in a thread save way.
/// @param[in] std::ostringstream().flush() << TEXT but used like print(TEXT) with macros
/// Aquires the printlock, prints the provided key and releases the printlock.
/// It flushes at the end without adding a newline character.
/// Note: This function is supossed to always be called using the print macro.
/// Example: print("Hallo " << name);
void printStream(std::ostream& os);

/// @brief Prints a stream in a thread save way and adds a newline at the end.
/// @param[in] std::ostringstream().flush() << TEXT but used like println(TEXT) with macros
/// Aquires the printlock, prints the provided key, adds a std::endl and
/// releases the printlock. It flushes with the std::endl.
/// Note: This function is supossed to always be called using the println macro.
/// Example: println("Hallo " << name);
void printlnStream(std::ostream& os);

/// @brief Converts a stream into an std::string
/// @param[in] std::ostringstream().flush() << TEXT but used like streamToString(TEXT) with macros
/// Note: This function is supossed to always be called using the streamToString macro.
/// Example: std::string greeting = streamToString("Hallo " << name);
/// @return Input stream converted as std::string
std::string convertStreamToString(std::ostream& os);


//#################//
//  CUDA KERNELS!  //
//   __global__    //
//#################//

#if defined(__NVCC__)

__global__ void cudaAssertValue(uint32_t * data);

/// @brief Calculates how much the IFFT output needs to be Y-shifted
/// @param[in] Hamming weight (amount of 1-bits) of toeplitz matrix seed
/// @param[in] Hamming weight (amount of 1-bits) of key
/// @param[out] The Kernels output will be stored in correction_float_dev
/// Because we use random 0 and 1 as input the average (without reduction) 
/// will always be around 0.5. This adds up as we have up to 2^27 as input
/// That would result in the first Element of the FFT (0 HZ) be around 2^26
/// After elementwise multiplication that would be 2^56 which leads to inaccurate
/// floating point precision. Because of that let's just zero out the first
/// element after FFT and correct with correction_float_dev after normalized IFFT
/// Simplified Calculation: arg3 = ((arg1 * arg2) / sample_size) % 2
__global__ void calculateCorrectionFloat(uint32_t* count_one_of_global_seed, uint32_t* count_one_of_global_key, float* correction_float_dev);
int unitTestCalculateCorrectionFloat();

/// @brief Sets the first Element of both input arrays to Zero.
/// @param[in,out] Output of key FFT
/// @param[in,out] Output of toeplitz seed FFT
/// It sets the first Element to float2(0.0f, 0.0f) which sets both the 
/// real and imaginary part to zero. This is done on GPU as it's faster
/// then acccessing that data from CPU as that way no PCIe datatransfer is needed.
/// This is done so for precision. See calculateCorrectionFloat for more information.
__global__ void setFirstElementToZero(Complex* do1, Complex* do2);
int unitTestSetFirstElementToZero();

/// @brief Calculates the element wise product of two Complex arrays
/// @param[in,out] Output of key FFT
/// @param[in] Output of toeplitz seed FFT
/// It stores the result in the same memory as the first input to save memory
/// Complex (float2) arrays from the size of 2^27 are 1074 MB each!
/// We only have limited GPU storage and the goal is to only require 8 GB.
/// Why the first? Because we already use the second one to store the result of the IFFT
/// and as precalculated FFT result if toeplitz matrix doesn't have to be recalulated
/// for the next block in which case the result of the IFFT is stored in di2.
__global__ void ElementWiseProduct(Complex* do1, Complex* do2);
int unitTestElementWiseProduct();

std::pair<double, double> FletcherFloat(float* data, int count);

bool isFletcherFloat(float* data, int count, const double expectedSum1, const double allowedAbsDeltaSum1, const double expectedSum2, const double allowedAbsDeltaSum2);

/// @brief Converts binary data to a float array where every bit represents one float
/// @param[in] Binary Input
/// @param[out] Floating point output
/// @param[in, out] Hamming weight (amount of 1-bits) (if initial value is 0)
/// This function also counts the Hamming weight (amount of 1-bits).
/// binary => float: 0 => 0.0f and 1 => 1.0f/reduction
__global__ void binInt2float(uint32_t* binIn, Real* realOut, uint32_t* count_one_global);
void unitTestBinInt2floatVerifyResultThread(float* floatOutTest, int i, int i_max);
int unitTestBinInt2float();
//David W. Wilson: https://oeis.org/A000788/a000788.txt
unsigned A000788(unsigned n);

/// @brief Generates the Privacy amplification results from the IFFT result and key rest 
/// @param[in] Raw IFFT output
/// @param[out] Privacy Amplification result in correct endianness
/// @param[in] Key rest with which the normalized converted IFFT result will be XORed
/// @param[in] Result from calculateCorrectionFloat
/// This function normalizes the IFFT floating point ouput and converts it to binary
/// while also XORing it with the key rest to optain the Privacy Amplification result.
__global__ void ToBinaryArray(Real* invOut, uint32_t* binOut, uint32_t* key_rest_local, Real* correction_float_dev);
void unitTestToBinaryArrayVerifyResultThread(uint32_t* binOutTest, uint32_t* key_rest_local, int i, int i_max);
int unitTestToBinaryArray();

#endif

/// @brief Prints bynary byte data.
/// @param position The memory location where the data to print start
/// @param end The memory location where the data to print end
/// Loops from start to end and prints each bytes binary representation
/// After printing all binary data a newline character is printed
void printBin(const uint8_t* position, const uint8_t* end);

/// @brief Prints binary integer data.
/// @param position The memory location where the data to print start
/// @param end The memory location where the data to print end
/// Loops from start to end and prints each integers binary representation
/// After printing all binary data a newline character is printed
void printBin(const uint32_t* position, const uint32_t* end);

/// @brief Splits the recived key into key_start and key_rest
/// This function also is responsible of filling dirty cache
/// regions with zeros the most efficient way possible.
/// The key rest has to be shifted by one bit which this function
/// also takes care of by using a memcopy loop that does exactly that.
/// Note: This function doesn't need any arguments because it uses 
/// key_start, key_rest, key_start_zero_pos, key_rest_zero_pos
/// input_cache_block_size and input_cache_write_pos
void key2StartRest();

/// @brief Reads one block of matrix seed data from a file.
/// This function reads from the toeplitz_seed_path path specified in config.yaml
/// The whole cache will be filled with that block so that
/// PrivacyAmplification will forever use that matrix seed.
/// See toeplitz_seed_path for more information.
void readMatrixSeedFromFile();

/// @brief Reads one block of key data from a file.
/// This function reads from the keyfile_path path specified in config.yaml
/// The whole cache will be filled with that block so that
/// PrivacyAmplification will forever use that key.
/// See keyfile_path for more information.
void readKeyFromFile();

/// @brief Recives data from the key server and matrix seed server or a file
/// This function contains all the code needed to communicate with
/// the key server and matrix seed server and reading it from a file.
/// This functions runs on its own Thread which is started inside main.
/// Because this function runs in parallel it never returns
/// 
/// Comunnication matrix seed server:
/// Action                    Size [bytes]                    endianness
/// Send: SYN                 3                               big
/// Receive: recv_key         ((sample_size / 32) * 4         4 byte little
/// 
/// Comunnication with key server:
/// Action                    Size [bytes]                    endianness
/// Send: SYN                 3                               big
/// Receive: vertical_block   4                               4 byte little
/// Receive: recv_key         ((sample_size / 32) + 1) * 4    4 byte little
void reciveData();

std::string toHexString(const unsigned char* data, uint32_t data_length);

bool isSha3(const unsigned char* dataToVerify, uint32_t dataToVerifySize, const uint8_t expectedHash[]);


/// @brief Prints binary byte data.
/// @param Data containing the Privacy Amplification result to verify
/// This function runs in a ThreadPool of the size verify_ampout_threads.
/// Warning: This function can fail if more tasks arrive than it can handle
/// because by that time the data it verifies would have already been overwritten
/// by the next trip though the output cache. This only occures if the thread pool
/// contains not enough threads or the output cache is too small. See verify_ampout,
/// verify_ampout_threads and output_blocks_to_cache for more information.
void verifyData(const unsigned char* dataToVerify);

/// @brief Sends data to the ReceiveAmpOutExample client or stores it in file
/// This function contains all the code needed to send the Privacy Amplification
/// result to ReceiveAmpOutExample client or storing it in a file.
/// This function runs on its own Thread which is started inside main.
/// Because this function runs in parallel it never returns
/// 
/// Comunnication with ReceiveAmpOutExample client:
/// Action                    Size [bytes]                    endianness
/// Receive: SYN               3                               big
/// Send: output_block        vertical_len / 8                4 byte little or big
/// If big endian is required, set AMPOUT_REVERSE_ENDIAN to true otherwise to false.
void sendData();

/// @brief Pharses config.yaml
/// If config.yaml doesn't exist the program to exit with error 102
/// If a value is missing in config.yaml the default value will be used instead.
void readConfig();

/// @brief Set the Windows consoles backgroud color and font color
void setConsoleDesign();

/// @brief Actual Privacy Amplification Algorithm
/// @param [UNUSED] Ammount of arguments
/// @param [UNUSED] Array of arguments
/// The main function contains the actual Privacy Amplification Algorithm
/// It handles and conrolls everything that happens in the main thread.
/// It also contains the whole memory management on both RAM and GPU.
int main(int argc, char* argv[]);
