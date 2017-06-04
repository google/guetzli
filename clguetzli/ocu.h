#pragma once

#ifdef __USE_CUDA__

#include <cuda.h>
#include "ocl.h"

#define LOG_CU_RESULT(e)   if (CUDA_SUCCESS != (e)) { LogError("Error: %s:%d returned %s.\n", __FUNCTION__, __LINE__, TranslateCUDAError((e)));}

struct ocu_args_d_t;

const char* TranslateCUDAError(CUresult errorCode);

ocu_args_d_t& getOcu(void);

struct ocu_args_d_t
{
    ocu_args_d_t();
    ~ocu_args_d_t();

    cu_mem allocMem(size_t s, const void *init = NULL);
    ocu_channels allocMemChannels(size_t s, const void *c0 = NULL, const void *c1 = NULL, const void *c2 = NULL);
    void releaseMemChannels(ocu_channels &rgb);

    CUfunction  kernel[KERNEL_COUNT];
    CUstream    commandQueue;
    CUmodule    mod;
    CUcontext   ctxt;
    CUdevice    dev;
};

#endif