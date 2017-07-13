/*
* Memory Pool for CUDA
*
* Author: ianhuang@tencent.com
*/
#pragma once

#ifdef __USE_CUDA__

#include <list>
#include <cuda.h>
#include "ocl.h"

/*Simple memory pool for CUDA, aiming to reduce the memory allocation count, because it's time consuming.*/

enum mem_block_status
{
    MBS_IDLE,
    MBS_BUSY,
};

struct cu_mem_block_t
{
    cu_mem_block_t()
        :status(MBS_IDLE)
        , used(0)
    {}
    ~cu_mem_block_t()
    {}

    mem_block_status status;
    size_t size;
    size_t used;
    cu_mem mem;
};

struct cu_mem_pool_t
{
    cu_mem_pool_t();
    ~cu_mem_pool_t();
    cu_mem allocMem(size_t s, const void *init = NULL);
    void releaseMem(cu_mem mem);
    void drain();

    std::list<cu_mem_block_t> mem_pool;
    CUstream    commandQueue;
    size_t alloc_count;
    size_t total_mem_request;
};

#endif