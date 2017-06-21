#include "cumem_pool.h"

#ifdef __USE_CUDA__

bool compare_size(const ocu_mem_block_t& first, const ocu_mem_block_t& second)
{
    return (first.size < second.size);
}

ocu_mem_pool_t::ocu_mem_pool_t()
    :alloc_count(0)
{

}

ocu_mem_pool_t::~ocu_mem_pool_t()
{

}

cu_mem ocu_mem_pool_t::allocMem(size_t s, const void *init)
{
    alloc_count++;
    ocu_mem_block_t *block_candidate = NULL;
    for (std::list<ocu_mem_block_t>::iterator iter = mem_pool.begin(); iter != mem_pool.end(); iter++)
    {
        ocu_mem_block_t *block = &(*iter);
        if (block->status == 0 && block->size >= s) {
            block_candidate = block;
            break;
        }
    }
    cu_mem mem = NULL;
    if (block_candidate != NULL) {
        block_candidate->status = 1;
        block_candidate->used = s;

        mem = block_candidate->mem;
        //LogError("mem_pool reuse mem:%lld, used:%lld.\r\n", block_candidate->size, block_candidate->used);
    }
    else {
        cu_mem new_mem;
        cuMemAlloc(&new_mem, s);
        ocu_mem_block_t mem_block;
        mem_block.size = s;
        mem_block.used = s;
        mem_block.mem = new_mem;
        mem_block.status = 1;
        mem_pool.push_back(mem_block);
        mem_pool.sort(compare_size);

        mem = new_mem;
        //LogError("mem_pool new mem:%lld, used:%lld.\r\n", mem_block.size, mem_block.used);
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

    //cu_mem mem;
    //cuMemAlloc(&mem, s);
    //if (init)
    //{
    //    cuMemcpyHtoDAsync(mem, init, s, commandQueue);
    //}
    //else
    //{
    //    cuMemsetD8Async(mem, 0, s, commandQueue);
    //}

    //return mem;
}

void ocu_mem_pool_t::releaseMem(cu_mem mem)
{
    ocu_mem_block_t *block_candidate = NULL;
    for (std::list<ocu_mem_block_t>::iterator iter = mem_pool.begin(); iter != mem_pool.end(); iter++)
    {
        ocu_mem_block_t *block = &(*iter);
        if (block->mem == mem) {
            block_candidate = block;
            break;
        }
    }
    if (block_candidate != NULL) {
        block_candidate->status = 0;
        block_candidate->used = 0;
    }
    else {
        cuMemFree(mem);
        LogError("mem_pool release mem:%lld can not be found.\r\n", mem);
    }

    //LogError("mem_pool release mem:%lld, used:%lld.\r\n", block_candidate->size, block_candidate->used);
}

void ocu_mem_pool_t::drain()
{
    size_t total_mem = 0;
    size_t total_block = mem_pool.size();
    ocu_mem_block_t *block_candidate = NULL;
    for (std::list<ocu_mem_block_t>::iterator iter = mem_pool.begin(); iter != mem_pool.end(); iter++)
    {
        if (iter->status == 0) {
            total_mem += iter->size;
            cuMemFree(iter->mem);
            iter = mem_pool.erase(iter);
        }
    }

    LogError("mem_pool has %u blocks, and total memory is:%f kb, total alloc count:%d.\r\n", total_block, (float)(total_mem) / 1024, alloc_count);
}

#endif