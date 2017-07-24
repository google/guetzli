/*
* CUDA Manager
*
* Author: strongtu@tencent.com
*/
#include "ocu.h"

#ifdef __USE_CUDA__
#include <cuda.h>
#include <nvrtc.h>

ocu_args_d_t& getOcu(void)
{
    static bool bInit = false;
    static ocu_args_d_t ocu;

    if (bInit == true) return ocu;

    bInit = true;

    CUresult err = cuInit(0);
    LOG_CU_RESULT(err);
    CUdevice dev = 0;
    CUcontext ctxt;
    CUstream  stream;

    err = cuCtxCreate(&ctxt, CU_CTX_SCHED_AUTO, dev);
    LOG_CU_RESULT(err);

    char name[1024];
    int proc_count = 0;
    int thread_count = 0;
    int cap_major = 0, cap_minor = 0;
    cuDeviceGetName(name, sizeof(name), dev);
    cuDeviceGetAttribute(&cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&proc_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&thread_count, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
    LogError("CUDA Adapter:%s Ver%d.%d MP %d MaxThread Per MP %d)\r\n", name, cap_major, cap_minor, proc_count, thread_count);

    char* ptx = nullptr;
    size_t src_size = 0;
if (sizeof(void*) == 8)
    ReadSourceFromFile("clguetzli/clguetzli.cu.ptx64", &ptx, &src_size);
else
    ReadSourceFromFile("clguetzli/clguetzli.cu.ptx32", &ptx, &src_size);

    CUmodule mod;
    CUjit_option jit_options[2];
    void *jit_optvals[2];
    jit_options[0] = CU_JIT_CACHE_MODE;
    jit_optvals[0] = (void*)(uintptr_t)CU_JIT_CACHE_OPTION_CA;
    err = cuModuleLoadDataEx(&mod, ptx, 1, jit_options, jit_optvals);
    LOG_CU_RESULT(err);

    delete[] ptx;

    cuModuleGetFunction(&ocu.kernel[KERNEL_CONVOLUTION], mod, "clConvolutionEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_CONVOLUTIONX], mod, "clConvolutionXEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_CONVOLUTIONY], mod, "clConvolutionYEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_SQUARESAMPLE], mod, "clSquareSampleEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_OPSINDYNAMICSIMAGE], mod, "clOpsinDynamicsImageEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_MASKHIGHINTENSITYCHANGE], mod, "clMaskHighIntensityChangeEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_EDGEDETECTOR], mod, "clEdgeDetectorMapEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_BLOCKDIFFMAP], mod, "clBlockDiffMapEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_EDGEDETECTORLOWFREQ], mod, "clEdgeDetectorLowFreqEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_DIFFPRECOMPUTE], mod, "clDiffPrecomputeEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_SCALEIMAGE], mod, "clScaleImageEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_AVERAGE5X5], mod, "clAverage5x5Ex");
    cuModuleGetFunction(&ocu.kernel[KERNEL_MINSQUAREVAL], mod, "clMinSquareValEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_DOMASK], mod, "clDoMaskEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_COMBINECHANNELS], mod, "clCombineChannelsEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_UPSAMPLESQUAREROOT], mod, "clUpsampleSquareRootEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_REMOVEBORDER], mod, "clRemoveBorderEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_ADDBORDER], mod, "clAddBorderEx");
    cuModuleGetFunction(&ocu.kernel[KERNEL_COMPUTEBLOCKZEROINGORDER], mod, "clComputeBlockZeroingOrderEx");

    cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED);
    cuCtxSetSharedMemConfig(CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE);

    cuStreamCreate(&stream, 0);

    ocu.dev = dev;
    ocu.commandQueue = stream;
    ocu.mod = mod;
    ocu.ctxt = ctxt;
    ocu.mem_pool.commandQueue = ocu.commandQueue;

    return ocu;
}

ocu_args_d_t::ocu_args_d_t()
    : dev(0)
    , commandQueue(NULL)
    , mod(NULL)
    , ctxt(NULL)
{

}

ocu_args_d_t::~ocu_args_d_t()
{
    cuModuleUnload(mod);
    cuCtxDestroy(ctxt);
    mem_pool.drain();
}

cu_mem ocu_args_d_t::allocMem(size_t s, const void *init)
{
    return mem_pool.allocMem(s, init);
}

void ocu_args_d_t::releaseMem(cu_mem mem)
{
    mem_pool.releaseMem(mem);
}

ocu_channels ocu_args_d_t::allocMemChannels(size_t s, const void *c0, const void *c1, const void *c2)
{
    const void *c[3] = { c0, c1, c2 };

    ocu_channels img;
    for (int i = 0; i < 3; i++)
    {
        img.ch[i] = allocMem(s, c[i]);
    }

    return img;
}

void ocu_args_d_t::releaseMemChannels(ocu_channels &rgb)
{
    for (int i = 0; i < 3; i++)
    {
        releaseMem(rgb.ch[i]);
        rgb.ch[i] = NULL;
    }
}

const char* TranslateCUDAError(CUresult errorCode)
{
    switch (errorCode)
    {
    case CUDA_SUCCESS: return "CUDA_SUCCESS";
    case CUDA_ERROR_INVALID_VALUE: return "CUDA_ERROR_INVALID_VALUE";
    case CUDA_ERROR_OUT_OF_MEMORY: return "CUDA_ERROR_OUT_OF_MEMORY";
    case CUDA_ERROR_NOT_INITIALIZED: return "CUDA_ERROR_NOT_INITIALIZED";
    case CUDA_ERROR_DEINITIALIZED: return "CUDA_ERROR_DEINITIALIZED";
    case CUDA_ERROR_PROFILER_DISABLED: return "CUDA_ERROR_PROFILER_DISABLED";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED: return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED: return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED: return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
    case CUDA_ERROR_NO_DEVICE: return "CUDA_ERROR_NO_DEVICE";
    case CUDA_ERROR_INVALID_DEVICE: return "CUDA_ERROR_INVALID_DEVICE";
    case CUDA_ERROR_INVALID_IMAGE: return "CUDA_ERROR_INVALID_IMAGE";
    case CUDA_ERROR_INVALID_CONTEXT: return "CUDA_ERROR_INVALID_CONTEXT";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
    case CUDA_ERROR_MAP_FAILED: return "CUDA_ERROR_MAP_FAILED";
    case CUDA_ERROR_UNMAP_FAILED: return "CUDA_ERROR_UNMAP_FAILED";
    case CUDA_ERROR_ARRAY_IS_MAPPED: return "CUDA_ERROR_ARRAY_IS_MAPPED";
    case CUDA_ERROR_ALREADY_MAPPED: return "CUDA_ERROR_ALREADY_MAPPED";
    case CUDA_ERROR_NO_BINARY_FOR_GPU: return "CUDA_ERROR_NO_BINARY_FOR_GPU";
    case CUDA_ERROR_ALREADY_ACQUIRED: return "CUDA_ERROR_ALREADY_ACQUIRED";
    case CUDA_ERROR_NOT_MAPPED: return "CUDA_ERROR_NOT_MAPPED";
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
    case CUDA_ERROR_ECC_UNCORRECTABLE: return "CUDA_ERROR_ECC_UNCORRECTABLE";
    case CUDA_ERROR_UNSUPPORTED_LIMIT: return "CUDA_ERROR_UNSUPPORTED_LIMIT";
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE: return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
    case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
    case CUDA_ERROR_INVALID_PTX: return "CUDA_ERROR_INVALID_PTX";
    case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT: return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
    // case CUDA_ERROR_NVLINK_UNCORRECTABLE: return "CUDA_ERROR_NVLINK_UNCORRECTABLE";
    case CUDA_ERROR_INVALID_SOURCE: return "CUDA_ERROR_INVALID_SOURCE";
    case CUDA_ERROR_FILE_NOT_FOUND: return "CUDA_ERROR_FILE_NOT_FOUND";
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
    case CUDA_ERROR_OPERATING_SYSTEM: return "CUDA_ERROR_OPERATING_SYSTEM";
    case CUDA_ERROR_INVALID_HANDLE: return "CUDA_ERROR_INVALID_HANDLE";
    case CUDA_ERROR_NOT_FOUND: return "CUDA_ERROR_NOT_FOUND";
    case CUDA_ERROR_NOT_READY: return "CUDA_ERROR_NOT_READY";
    case CUDA_ERROR_ILLEGAL_ADDRESS: return "CUDA_ERROR_ILLEGAL_ADDRESS";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
    case CUDA_ERROR_LAUNCH_TIMEOUT: return "CUDA_ERROR_LAUNCH_TIMEOUT";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
    case CUDA_ERROR_CONTEXT_IS_DESTROYED: return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
    case CUDA_ERROR_ASSERT: return "CUDA_ERROR_ASSERT";
    case CUDA_ERROR_TOO_MANY_PEERS: return "CUDA_ERROR_TOO_MANY_PEERS";
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
    case CUDA_ERROR_HARDWARE_STACK_ERROR: return "CUDA_ERROR_HARDWARE_STACK_ERROR";
    case CUDA_ERROR_ILLEGAL_INSTRUCTION: return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
    case CUDA_ERROR_MISALIGNED_ADDRESS: return "CUDA_ERROR_MISALIGNED_ADDRESS";
    case CUDA_ERROR_INVALID_ADDRESS_SPACE: return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
    case CUDA_ERROR_INVALID_PC: return "CUDA_ERROR_INVALID_PC";
    case CUDA_ERROR_LAUNCH_FAILED: return "CUDA_ERROR_LAUNCH_FAILED";
    case CUDA_ERROR_NOT_PERMITTED: return "CUDA_ERROR_NOT_PERMITTED";
    case CUDA_ERROR_NOT_SUPPORTED: return "CUDA_ERROR_NOT_SUPPORTED";
    case CUDA_ERROR_UNKNOWN: return "CUDA_ERROR_UNKNOWN";
    default: return "CUDA_ERROR_UNKNOWN";
    }
}
#endif