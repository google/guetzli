#pragma once

#include <cuda.h>
#include "ocl.h"

struct ocu_args_d_t;

ocu_args_d_t& getOcu(void);

struct ocu_args_d_t
{
    ocu_args_d_t();
    ~ocu_args_d_t();

    CUdeviceptr allocMem(size_t s, const void *init = NULL);
    ocu_channels allocMemChannels(size_t s, const void *c0 = NULL, const void *c1 = NULL, const void *c2 = NULL);
    void releaseMemChannels(ocu_channels &rgb);

    CUfunction  kernel[KERNEL_COUNT];
    CUstream    stream;
    CUmodule    mod;
    CUcontext   ctxt;
    CUdevice    dev;
};