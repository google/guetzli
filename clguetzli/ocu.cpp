
#include <cuda.h>
#include "ocu.h"

ocu_args_d_t& getOcu(void)
{
    static bool bInit = false;
    static ocu_args_d_t ocu;

    if (bInit == true) return ocu;

    cuInit(0);

    CUresult r;
    CUcontext ctxt;
    CUdevice dev = 0;

    cuCtxCreate(&ctxt, CU_CTX_SCHED_BLOCKING_SYNC, dev);

    char name[1024];
    int proc_count = 0;
    int thread_count = 0;
    int cap_major = 0, cap_minor = 0;
    cuDeviceGetName(name, sizeof(name), dev);
    cuDeviceGetAttribute(&cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&proc_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&thread_count, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
    LogError("CUDA Adapter:%s Ver%d.%d (%d x %d)\r\n", name, cap_major, cap_minor, proc_count, thread_count);

    CUmodule mod;

    char* source = nullptr;
    size_t src_size = 0;
    ReadSourceFromFile("clguetzli/clguetzli.cu.ptx30", &source, &src_size);

    CUjit_option jit_options[2];
    void *jit_optvals[2];
    jit_options[0] = CU_JIT_CACHE_MODE;
    jit_optvals[0] = (void*)(uintptr_t)CU_JIT_CACHE_OPTION_CA;
    cuModuleLoadDataEx(&mod, source, 1, jit_options, jit_optvals);

    delete[] source;

    cuModuleGetFunction(&ocu.kernel[KERNEL_SCALEIMAGE], mod, "clScaleImageEx");

    cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED);
    cuCtxSetSharedMemConfig(CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE);

    cuStreamCreate(&ocu.stream, 0);

    return ocu;
}

ocu_args_d_t::ocu_args_d_t()
{

}

ocu_args_d_t::~ocu_args_d_t()
{

}

CUdeviceptr ocu_args_d_t::allocMem(size_t s, const void *init)
{
    CUdeviceptr mem;
    cuMemAlloc(&mem, s);
    if (init)
    {
        cuMemcpyHtoDAsync(mem, init, s, this->stream);
    }
    else
    {
        cuMemsetD8(mem, 0, s);
    }

    return mem;
}