#include <cuda.h>
#include <nvrtc.h>
#include "ocu.h"

ocu_args_d_t& getOcu(void)
{
    static bool bInit = false;
    static ocu_args_d_t ocu;

    if (bInit == true) return ocu;

    bInit = true;

    CUresult r = cuInit(0);
    CUdevice dev = 0;
    CUcontext ctxt;
    CUstream  stream;

    r = cuCtxCreate(&ctxt, CU_CTX_SCHED_BLOCKING_SYNC, dev);

    char name[1024];
    int proc_count = 0;
    int thread_count = 0;
    int cap_major = 0, cap_minor = 0;
    cuDeviceGetName(name, sizeof(name), dev);
    cuDeviceGetAttribute(&cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&proc_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&thread_count, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
    LogError("CUDA Adapter:%s Ver%d.%d MP %d Core %d)\r\n", name, cap_major, cap_minor, proc_count, thread_count);

    char* source = nullptr;
    size_t src_size = 0;
    ReadSourceFromFile("clguetzli/clguetzli.cl", &source, &src_size);

    nvrtcProgram prog;
    const char *opts[] = { "-arch=compute_30", "-default-device", "-G", "-I\"./\"", "--fmad=false" };
    nvrtcCreateProgram(&prog, source, "clguetzli.cl", 0, NULL, NULL);
    nvrtcResult compile_result = nvrtcCompileProgram(prog, 3, opts);
    if (NVRTC_SUCCESS != compile_result)
    {
        // Obtain compilation log from the program.
        size_t logSize = 0;
        nvrtcGetProgramLogSize(prog, &logSize);
        char *log = new char[logSize];
        nvrtcGetProgramLog(prog, log);

        LogError("BuildInfo:\r\n%s\r\n", log);

        delete[] log;
    }

    // Obtain PTX from the program.
    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    char *ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    CUmodule mod;
    CUjit_option jit_options[2];
    void *jit_optvals[2];
    jit_options[0] = CU_JIT_CACHE_MODE;
    jit_optvals[0] = (void*)(uintptr_t)CU_JIT_CACHE_OPTION_CA;
    r = cuModuleLoadDataEx(&mod, ptx, 1, jit_options, jit_optvals);

    delete[] source;
    delete[] ptx;

    r = cuModuleGetFunction(&ocu.kernel[KERNEL_CONVOLUTION], mod, "clConvolutionEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_CONVOLUTIONX], mod, "clConvolutionXEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_CONVOLUTIONY], mod, "clConvolutionYEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_SQUARESAMPLE], mod, "clSquareSampleEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_OPSINDYNAMICSIMAGE], mod, "clOpsinDynamicsImageEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_MASKHIGHINTENSITYCHANGE], mod, "clMaskHighIntensityChangeEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_EDGEDETECTOR], mod, "clEdgeDetectorMapEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_BLOCKDIFFMAP], mod, "clBlockDiffMapEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_EDGEDETECTORLOWFREQ], mod, "clEdgeDetectorLowFreqEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_DIFFPRECOMPUTE], mod, "clDiffPrecomputeEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_SCALEIMAGE], mod, "clScaleImageEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_AVERAGE5X5], mod, "clAverage5x5Ex");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_MINSQUAREVAL], mod, "clMinSquareValEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_DOMASK], mod, "clDoMaskEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_COMBINECHANNELS], mod, "clCombineChannelsEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_UPSAMPLESQUAREROOT], mod, "clUpsampleSquareRootEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_REMOVEBORDER], mod, "clRemoveBorderEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_ADDBORDER], mod, "clAddBorderEx");
    r = cuModuleGetFunction(&ocu.kernel[KERNEL_COMPUTEBLOCKZEROINGORDER], mod, "clComputeBlockZeroingOrderEx");

    cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED);
    cuCtxSetSharedMemConfig(CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE);

    cuStreamCreate(&stream, 0);

    ocu.dev = dev;
    ocu.stream = stream;
    ocu.mod = mod;
    ocu.ctxt = ctxt;

    return ocu;
}

ocu_args_d_t::ocu_args_d_t()
{

}

ocu_args_d_t::~ocu_args_d_t()
{
    cuModuleUnload(mod);
    cuCtxDestroy(ctxt);
//    cuStreamDestroy(stream);
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