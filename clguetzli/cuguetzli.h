#pragma once
#include "guetzli/processor.h"
#include "clguetzli.cl.h"

void cuOpsinDynamicsImage(float *r, float *g, float *b, const size_t xsize, const size_t ysize);

void cuDiffmapOpsinDynamicsImage(
    float* result,
    const float* r, const float* g, const float* b,
    const float* r2, const float* g2, const float* b2,
    const size_t xsize, const size_t ysize,
    const size_t step);

void cuComputeBlockZeroingOrder(
    guetzli::CoeffData *output_order_batch,
    const channel_info orig_channel[3],
    const float *orig_image_batch,
    const float *mask_scale,
    const int image_width,
    const int image_height,
    const channel_info mayout_channel[3],
    const int factor,
    const int comp_mask,
    const float BlockErrorLimit);

void cuMask(
    float* mask_r, float* mask_g, float* mask_b,
    float* maskdc_r, float* maskdc_g, float* maskdc_b,
    const size_t xsize, const size_t ysize,
    const float* r, const float* g, const float* b,
    const float* r2, const float* g2, const float* b2);

void cuBlurEx(CUdeviceptr image/*out, opt*/, const size_t xsize, const size_t ysize,
    const double sigma, const double border_ratio,
    CUdeviceptr result = NULL/*out, opt*/);

void cuOpsinDynamicsImageEx(ocu_channels &rgb, const size_t xsize, const size_t ysize);