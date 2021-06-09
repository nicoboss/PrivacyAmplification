#pragma once

namespace vuda
{
    //__host__ ​ __device__ 
    inline error_t free(void* devPtr)
    {
        //
        // get device assigned to thread
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(std::this_thread::get_id());

        //
        // free the allocation on the device
        tinfo->GetLogicalDevice()->free(devPtr);

        return vudaSuccess;
    }

    // Frees page - locked memory.
    /*__host__*/
    inline error_t freeHost(void* ptr)
    {
        return free(ptr);
    }

    //Allocates page-locked memory on the host.
    /*__host__*/
    inline error_t hostAlloc(void** pHost, size_t size, unsigned int flags, bool aligned)
    {
        //
        // get device assigned to thread
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(std::this_thread::get_id());

        //
        // allocate mem on the device
        if(flags == hostAllocDefault)
            tinfo->GetLogicalDevice()->mallocHost(pHost, size, aligned);
        /*else if(flags == hostAllocPortable)
        else if(flags == hostAllocMapped)*/
        else if(flags == hostAllocWriteCombined)
            tinfo->GetLogicalDevice()->hostAlloc(pHost, size, aligned);
        else
            assert(0);

        return vudaSuccess;
    }

    // __host__ ​ __device__
    inline error_t malloc(void** devPtr, size_t size, bool aligned)
    {
        //
        // get device assigned to thread
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(std::this_thread::get_id());
        
        //
        // allocate mem on the device
        tinfo->GetLogicalDevice()->malloc(devPtr, size, aligned);

        return vudaSuccess;
    }

    // Allocates page - locked memory on the host.
    /*__host__*/
    inline error_t mallocHost(void** ptr, size_t size, bool aligned)
    {
        //
        // get device assigned to thread
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(std::this_thread::get_id());

        //
        // allocate mem on the device
        tinfo->GetLogicalDevice()->mallocHost(ptr, size, aligned);

        return vudaSuccess;
    }

    /**
     * __host__
     * Copies count bytes from the memory area pointed to by src to the memory area pointed to by dst,
     * where kind specifies the direction of the copy, and must be one of
     * ::memcpyHostToHost, ::memcpyHostToDevice, ::memcpyDeviceToHost, ::memcpyDeviceToDevice, or ::memcpyDefault.
     */
    inline error_t memcpy(void* dst, const void* src, const size_t count, const memcpyKind kind, const stream_t stream=0)
    {
        const std::thread::id tid = std::this_thread::get_id();

        //
        // get device assigned to thread
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(tid);

        if(kind == vuda::memcpyHostToHost)
        {
            //
            // [ consider introducing internal synchronization with device queues (might work better with async jobs) ]
            std::memcpy(dst, src, count);
            //tinfo.GetLogicalDevice()->memcpyHtH(tid, dst, src, count, stream);
        }
        else if(kind == vuda::memcpyHostToDevice)
        {
            //
            // submit the copy command to command buffer
            tinfo->GetLogicalDevice()->memcpyToDevice(tid, dst, src, count, stream);
        }
        else if(kind == vuda::memcpyDeviceToDevice)
        {
            //
            // submit the copy command to command buffer
            tinfo->GetLogicalDevice()->memcpyDeviceToDevice(tid, dst, src, count, stream);
        }
        else if(kind == vuda::memcpyDeviceToHost)
        {
            //
            // submit the copy command to command buffer
            tinfo->GetLogicalDevice()->memcpyToHost(tid, dst, src, count, stream);
        }

        return vudaSuccess;
    }

    /*__host__*/
    inline error_t memset(void* devPtr, int value, size_t count)
    {
        const std::thread::id tid = std::this_thread::get_id();

        //
        // get device assigned to thread
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(std::this_thread::get_id());

        //
        // allocate mem on the device
        tinfo->GetLogicalDevice()->memset(tid, devPtr, value, count);

        return vudaSuccess;
    }

    /*__host__*/
    inline error_t memsetAsync(void* devPtr, int value, size_t count, const stream_t stream)
    {
        const std::thread::id tid = std::this_thread::get_id();

        //
        // get device assigned to thread
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(std::this_thread::get_id());

        //
        // allocate mem on the device
        tinfo->GetLogicalDevice()->memsetAsync(tid, devPtr, value, count, stream);

        return vudaSuccess;
    }

} //namespace vuda