#include <iostream>
#include <iomanip>
#include <assert.h>
#include <iterator>
#include <math.h>
#include <iostream>
#include <sstream>
#include "yaml/Yaml.hpp"


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
#define TEST


string address_seed_in;
string address_key_in;
string address_amp_out;
int32_t* reuse_seed_amount_array;

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
bool do_xor_key_rest = true;


atomic<uint32_t> input_cache_read_pos_seed;
atomic<uint32_t> input_cache_read_pos_key;
atomic<uint32_t> input_cache_write_pos_seed;
atomic<uint32_t> input_cache_write_pos_key;
atomic<uint32_t> output_cache_read_pos;
atomic<uint32_t> output_cache_write_pos;
mutex printlock;
float normalisation_float;
atomic<bool> unitTestsFailed = false;
atomic<bool> unitTestBinInt2floatVerifyResultThreadFailed = false;
atomic<bool> unitTestToBinaryArrayVerifyResultThreadFailed = false;
atomic<bool> cuFFT_planned = false;


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

	reuse_seed_amount = root["reuse_seed_amount"].As<int32_t>(0);
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
	reuse_seed_amount_array = (int32_t*)calloc(input_blocks_to_cache, sizeof(int32_t));
	recv_key = (uint32_t*)malloc(key_blocks * sizeof(uint32_t));
	key_start_zero_pos = (uint32_t*)malloc(input_blocks_to_cache * sizeof(uint32_t));
	key_rest_zero_pos = (uint32_t*)malloc(input_blocks_to_cache * sizeof(uint32_t));
}


int main(int argc, char* argv[])
{
	//About
	string about = streamToString("# PrivacyAmplification v" << VERSION << " by Nico Bosshard from " << __DATE__ << " #");
	string border(about.length(), '#');
	cout << border << endl << about << endl << border << endl << endl;
	readConfig();

	cout << "#PrivacyAmplification with " << sample_size << " bits" << endl << endl;

	cudaSetDevice(0); //cudaSetDevice(cuda_device_id_to_use);

	Real* di1; //Device Input 1
	Real* di2; //Device Input 2
	Real* invOut;  //Result of the IFFT (uses the same memory as do2)

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

	glslang_initialize_process();

	cudaMallocHost((void**)&testMemoryHost, max(sample_size * sizeof(Complex), (sample_size + 992) * sizeof(Real)));

	float* hi;
	cudaMalloc((void**)&hi, sizeof(float));


	cudaMalloc((void**)&di1, (uint64_t)sizeof(float) * 2 * ((sample_size + 992) / 2 + 1));
	println(new VkBuffer{ logical_device->GetBuffer(di1) });
	cudaMalloc((void**)&di2, (sample_size + 992) * sizeof(Real));
	println(new VkBuffer{ logical_device->GetBuffer(di1) });

	VkFFTConfiguration configuration = {};
	VkFFTApplication app = {};
	configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
	configuration.size[0] = 16384; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
	configuration.size[1] = 1;
	configuration.size[2] = 1;

	configuration.aimThreads = 1024;
	configuration.registerBoost = true;
	configuration.performHalfBandwidthBoost = true;
	configuration.useLUT = false;
	configuration.normalize = false;

	configuration.performR2C = true; //Perform R2C/C2R transform. Can be combined with all other options. Reduces memory requirements by a factor of 2. Requires special input data alignment: for x*y*z system pad x*y plane to (x+2)*y with last 2*y elements reserved, total array dimensions are (x*y+2y)*z. Memory layout after R2C and before C2R can be found on github.
	//configuration.disableMergeSequencesR2C = 1;
	//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
	configuration.device = &vkGPU.device;
	configuration.queue = &vkGPU.queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
	configuration.fence = &vkGPU.fence;
	configuration.commandPool = &vkGPU.commandPool;
	configuration.physicalDevice = &vkGPU.physicalDevice;
	configuration.isCompilerInitialized = true;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

	VkBuffer a = logical_device->GetBuffer(di1);
	configuration.buffer = &a;
	bufferSize = (uint64_t)sizeof(float) * 2 * (sample_size / 2 + 1);
	configuration.bufferSize = &bufferSize;

	resFFT = initializeVkFFT(&app, configuration);
	if (resFFT != VKFFT_SUCCESS) return resFFT;

	//##########################
	// Mainloop of main thread #
	//##########################
	while (true) {

		cudaMemset(di1, 0x3f800000, sample_size * sizeof(uint32_t));

		cudaMemcpy(testMemoryHost, di1, 2 * (sample_size / 2 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < 100; i += 2) {
			println(i << "[IN]" << ": " << reinterpret_cast<float*>(testMemoryHost)[i] << "|" << reinterpret_cast<float*>(testMemoryHost)[i + 1]);
		}

		vkfftExecR2C(&vkGPU, &app);
		cudaStreamSynchronize(0);

		cudaMemcpy(testMemoryHost, di1, 2 * (sample_size / 2 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < 100; i += 2) {
			println(i << "[OUT]"  << ": " << reinterpret_cast<float*>(testMemoryHost)[i] << "|" << reinterpret_cast<float*>(testMemoryHost)[i + 1]);
		}

	}

	// Deallocate memoriey on GPU and RAM
	cudaFree(di1);
	cudaFree(di2);
	cudaFree(invOut);
	cudaFree(Output);
	cudaFree(testMemoryHost);
	return 0;
}
