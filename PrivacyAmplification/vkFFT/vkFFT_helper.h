#pragma once

#include <vector>
#if(VKFFT_BACKEND==0)
#include "vulkan/vulkan.h"
#elif(VKFFT_BACKEND==1)
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#elif(VKFFT_BACKEND==2)
#define __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#endif
#include "vkFFT.h"

const bool enableValidationLayers = false;

typedef struct {
#if(VKFFT_BACKEND==0)
	VkInstance instance;//a connection between the application and the Vulkan library 
	VkPhysicalDevice physicalDevice;//a handle for the graphics card used in the application
	VkPhysicalDeviceProperties physicalDeviceProperties;//bastic device properties
	VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;//bastic memory properties of the device
	VkDevice device;//a logical device, interacting with physical device
	VkDebugUtilsMessengerEXT debugMessenger;//extension for debugging
	uint64_t queueFamilyIndex;//if multiple queues are available, specify the used one
	VkQueue queue;//a place, where all operations are submitted
	VkCommandPool commandPool;//an opaque objects that command buffer memory is allocated from
	VkFence fence;//a vkGPU->fence used to synchronize dispatches
	std::vector<const char*> enabledDeviceExtensions;
#elif(VKFFT_BACKEND==1)
	CUdevice device;
	CUcontext context;
#elif(VKFFT_BACKEND==2)
	hipDevice_t device;
	hipCtx_t context;
#elif(VKFFT_BACKEND==3)
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue commandQueue;
#endif
	uint64_t device_id;//an id of a device, reported by Vulkan device list
} VkGPU;//an example structure containing Vulkan primitives

VkFFTResult performVulkanFFT(VkGPU* vkGPU, VkFFTApplication* app, VkFFTLaunchParams* launchParams, int inverse) {
	VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferAllocateInfo.commandPool = vkGPU->commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	res = vkAllocateCommandBuffers(vkGPU->device, &commandBufferAllocateInfo, &commandBuffer);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	res = vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;
	launchParams->commandBuffer = &commandBuffer;
	resFFT = VkFFTAppend(app, inverse, launchParams);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	res = vkEndCommandBuffer(commandBuffer);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	res = vkQueueSubmit(vkGPU->queue, 1, &submitInfo, vkGPU->fence);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
	res = vkWaitForFences(vkGPU->device, 1, &vkGPU->fence, VK_TRUE, 100000000000);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
	res = vkResetFences(vkGPU->device, 1, &vkGPU->fence);
	if (res != 0) return VKFFT_ERROR_FAILED_TO_RESET_FENCES;
	vkFreeCommandBuffers(vkGPU->device, vkGPU->commandPool, 1, &commandBuffer);
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
	resFFT = VkFFTAppend(app, inverse, launchParams);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	res = cudaDeviceSynchronize();
	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	resFFT = VkFFTAppend(app, inverse, launchParams);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	res = hipDeviceSynchronize();
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
	launchParams->commandQueue = &vkGPU->commandQueue;
	resFFT = VkFFTAppend(app, inverse, launchParams);
	if (resFFT != VKFFT_SUCCESS) return resFFT;
	res = clFinish(vkGPU->commandQueue);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
#endif
	return resFFT;
}

VkFFTResult vkfftExecR2C(VkGPU* vkGPU, VkFFTApplication* app) {
	VkFFTLaunchParams launchParams = {};
	return performVulkanFFT(vkGPU, app, &launchParams, -1);
}

VkFFTResult vkfftExecC2R(VkGPU* vkGPU, VkFFTApplication* app) {
	VkFFTLaunchParams launchParams = {};
	return performVulkanFFT(vkGPU, app, &launchParams, 1);
}
