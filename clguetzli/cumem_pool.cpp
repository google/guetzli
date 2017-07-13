/*
 * Memory Pool for CUDA
 *
 * Author: ianhuang@tencent.com
 */

#include "cumem_pool.h"

#ifdef __USE_CUDA__

bool compare_size(const cu_mem_block_t& first, const cu_mem_block_t& second)
{
    return (first.size < second.size);
}

cu_mem_pool_t::cu_mem_pool_t()
    : alloc_count(0)
    , total_mem_request(0)
{

}

cu_mem_pool_t::~cu_mem_pool_t()
{

}

cu_mem cu_mem_pool_t::allocMem(size_t s, const void *init)
{
    alloc_count++;
    total_mem_request += s;
    cu_mem_block_t *block_candidate = NULL;
    for (std::list<cu_mem_block_t>::iterator iter = mem_pool.begin(); iter != mem_pool.end(); iter++)
    {
        cu_mem_block_t *block = &(*iter);
        if (block->status == MBS_IDLE && block->size >= s) {
            block_candidate = block;
            break;
        }
    }
    cu_mem mem = NULL;
    if (block_candidate != NULL) {
        block_candidate->status = MBS_BUSY;
        block_candidate->used = s;

        mem = block_candidate->mem;
    }
    else {
        cu_mem new_mem;
        cuMemAlloc(&new_mem, s);
        cu_mem_block_t mem_block;
        mem_block.size = s;
        mem_block.used = s;
        mem_block.mem = new_mem;
        mem_block.status = MBS_BUSY;
        mem_pool.push_back(mem_block);
        mem_pool.sort(compare_size);

        mem = new_mem;
    }
    if (init)
    {
        cuMemcpyHtoDAsync(mem, init, s, commandQueue);
    }
    else
    {
        cuMemsetD8Async(mem, 0, s, commandQueue);
    }

    return mem;
}

void cu_mem_pool_t::releaseMem(cu_mem mem)
{
    cu_mem_block_t *block_candidate = NULL;
    for (std::list<cu_mem_block_t>::iterator iter = mem_pool.begin(); iter != mem_pool.end(); iter++)
    {
        cu_mem_block_t *block = &(*iter);
        if (block->mem == mem) {
            block_candidate = block;
            break;
        }
    }
    if (block_candidate != NULL) {
        block_candidate->status = MBS_IDLE;
        block_candidate->used = 0;
    }
    else {
        cuMemFree(mem);
        LogError("mem_pool release mem:%lld can not be found.\r\n", mem);
    }
}

void cu_mem_pool_t::drain()
{
    size_t total_mem = 0;
    size_t total_block = mem_pool.size();
    cu_mem_block_t *block_candidate = NULL;
    for (std::list<cu_mem_block_t>::iterator iter = mem_pool.begin(); iter != mem_pool.end(); iter++)
    {
        if (iter->status == MBS_IDLE) {
            total_mem += iter->size;
            cuMemFree(iter->mem);
            iter = mem_pool.erase(iter);
        }
    }

    LogError("mem_pool has %u blocks, and total pool memory is:%f kb, total memory request:%f kb, total alloc count:%d.\r\n", total_block, (float)(total_mem) / 1024, (float)(total_mem_request) / 1024, alloc_count);
}

#endif