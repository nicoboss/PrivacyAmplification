#pragma once

namespace vuda
{
    namespace detail
    {
        //
        // singleton interface for vulkan logical devices
        //
        class interface_logical_devices final : public singleton
        {
        public:

            static logical_device* create(const vk::PhysicalDevice& physDevice, const int device, uint32_t vudaChunkSize)
            {
                std::unordered_map<int, logical_device>::iterator iter;
                
                //
                // take exclusive lock
                // [ contention ]
                std::lock_guard<std::shared_mutex> lck(mtx());

                //
                // return the logical device if it exists
                iter = get().find(device);
                if(iter != get().end())
                    return &iter->second;
                    
                //
                // create a logical device if it is not already in existence
                return create_logical_device(physDevice, device, vudaChunkSize);
            }

        private:

            static std::unordered_map<int, logical_device>& get(void)
            {
                static std::unordered_map<int, logical_device> local_logical_devices;
                return local_logical_devices;
            }

            static std::shared_mutex& mtx(void)
            {
                static std::shared_mutex local_mtx;
                return local_mtx;
            }

            /*static std::atomic_flag& writers_lock_af(void)
            {
                static std::atomic_flag wlock = ATOMIC_FLAG_INIT;
                return wlock;
            }

            static std::atomic<bool>& writers_lock(void)
            {
                static std::atomic<bool> wlock = false;
                return wlock;
            }

            static std::condition_variable_any& cv(void)
            {
                static std::condition_variable_any cv;
                return cv;
            }*/

            static logical_device* create_logical_device(const vk::PhysicalDevice& physDevice, const int device, uint32_t vudaChunkSize)
            {
                //
                // get physical device
                //const vk::PhysicalDevice& physDevice = Instance::GetPhysicalDevice(device);

                //
                // get the QueueFamilyProperties of the PhysicalDevice
                std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physDevice.getQueueFamilyProperties();

                //
                // get the first index into queueFamiliyProperties which supports compute
                uint32_t computeQueueFamilyIndex = detail::vudaGetFamilyQueueIndex(queueFamilyProperties, vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer);

                //
                // HARDCODED MAX NUMBER OF STREAMS
                const uint32_t queueCount = queueFamilyProperties[computeQueueFamilyIndex].queueCount;
                const uint32_t queueComputeCount = queueCount;
                const std::vector<float> queuePriority(queueComputeCount, 0.0f);
                vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), computeQueueFamilyIndex, queueComputeCount, queuePriority.data());

            #ifdef VUDA_STD_LAYER_ENABLED
                vk::DeviceCreateInfo info(vk::DeviceCreateFlags(), 1, &deviceQueueCreateInfo, 1, Instance::vk_validationLayers.data(), 0, nullptr, nullptr);
            #else
                vk::DeviceCreateInfo info(vk::DeviceCreateFlags(), 1, &deviceQueueCreateInfo);
            #endif

                VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
                //queueCreateInfo.queueFamilyIndex = vkGPU->queueFamilyIndex;
                queueCreateInfo.queueCount = 1;
                float queuePriorities = 1.0;
                queueCreateInfo.pQueuePriorities = &queuePriorities;
                VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
                VkPhysicalDeviceFeatures deviceFeatures = {};
                //deviceCreateInfo.enabledExtensionCount = vkGPU->enabledDeviceExtensions.size();
                //deviceCreateInfo.ppEnabledExtensionNames = vkGPU->enabledDeviceExtensions.data();
                deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
                deviceCreateInfo.queueCreateInfoCount = 1;
                deviceCreateInfo.pEnabledFeatures = NULL;
                deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

                //get().insert({ device, logical_device(info, physDevice) });
                //auto pair = get().emplace(std::piecewise_construct, std::forward_as_tuple(device), std::forward_as_tuple(info, physDevice));
                // c++17
                auto pair = get().try_emplace(device, deviceCreateInfo, physDevice, vudaChunkSize);
                assert(pair.second);
                return &pair.first->second;
            }
        };

    } //namespace detail
} //namespace vuda