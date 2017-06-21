#pragma once

#ifdef __USE_CUDA__

#include <list>
#include <cuda.h>
#include "ocl.h"

struct ocu_mem_block_t
{
    ocu_mem_block_t()
        :status(0)
        , used(0)
    {}
    ~ocu_mem_block_t()
    {}

    int status;
    size_t size;
    size_t used;
    cu_mem mem;
};

struct ocu_mem_pool_t
{
    ocu_mem_pool_t();
    ~ocu_mem_pool_t();
    cu_mem allocMem(size_t s, const void *init = NULL);
    void releaseMem(cu_mem mem);
    void drain();

    std::list<ocu_mem_block_t> mem_pool;
    CUstream    commandQueue;
    size_t alloc_count;
};

#endif