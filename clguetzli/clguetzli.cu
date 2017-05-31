__global__ void clScaleImageEx(float *img, double scale)
{
    const int i = blockIdx.x;
    img[i] *= scale;
}