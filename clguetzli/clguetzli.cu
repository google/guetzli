#include "clguetzli\clguetzli.cl.h"

#ifdef __CUDACC__
//#ifdef __OPENCL_VERSION__
__device__ int get_global_id(int dim)
{
    switch (dim)
    {
    case 0:
        return threadIdx.x;
    case 1:
        return threadIdx.y;
    case 2:
        return threadIdx.z;
    default:
        return threadIdx.x;
    }
}
#endif


extern "C" __global__ void clScaleImageEx(float * img, double scale)
{
    const int i = get_global_id(0);
    img[i] = 0.0001;
}
