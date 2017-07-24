/*
* CUDA edition implementation of guetzli.
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*         chriskzhou@tencent.com
*/
#pragma once
#include "guetzli/processor.h"
#include "clguetzli.cl.h"
#include "ocu.h"

#ifdef __USE_CUDA__

#ifdef __USE_DOUBLE_AS_FLOAT__
#define double float
#endif

void cuOpsinDynamicsImage(
	float *r, float *g, float *b, 
	const size_t xsize, const size_t ysize);

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

void cuDiffmapOpsinDynamicsImageEx(
    cu_mem result,
    ocu_channels xyb0,
    ocu_channels xyb1,
    const size_t xsize, const size_t ysize,
    const size_t step);

void cuConvolutionXEx(
    cu_mem result/*out*/,
    const cu_mem inp, size_t xsize, size_t ysize,
    const cu_mem multipliers, size_t len,
    int xstep, int offset, float border_ratio);

void cuConvolutionYEx(
    cu_mem result/*out*/,
    const cu_mem inp, size_t xsize, size_t ysize,
    const cu_mem multipliers, size_t len,
    int xstep, int offset, float border_ratio);

void cuSquareSampleEx(
    cu_mem result/*out*/,
    const cu_mem image, size_t xsize, size_t ysize,
    size_t xstep, size_t ystep);

void cuBlurEx(cu_mem image/*out, opt*/, const size_t xsize, const size_t ysize,
    const double sigma, const double border_ratio,
    cu_mem result = NULL/*out, opt*/);

void cuOpsinDynamicsImageEx(ocu_channels &rgb, const size_t xsize, const size_t ysize);

void cuMaskHighIntensityChangeEx(
    ocu_channels &xyb0/*in,out*/,
    ocu_channels &xyb1/*in,out*/,
    const size_t xsize, const size_t ysize);

void cuEdgeDetectorMapEx(
    cu_mem result/*out*/,
    const ocu_channels &rgb, const ocu_channels &rgb2,
    const size_t xsize, const size_t ysize, const size_t step);

void cuBlockDiffMapEx(
    cu_mem block_diff_dc/*out*/,
    cu_mem block_diff_ac/*out*/,
    const ocu_channels &rgb, const ocu_channels &rgb2,
    const size_t xsize, const size_t ysize, const size_t step);

void cuEdgeDetectorLowFreqEx(
    cu_mem block_diff_ac/*in,out*/,
    const ocu_channels &rgb, const ocu_channels &rgb2,
    const size_t xsize, const size_t ysize, const size_t step);

void cuDiffPrecomputeEx(
    ocu_channels &mask/*out*/,
    const ocu_channels &xyb0, const ocu_channels &xyb1,
    const size_t xsize, const size_t ysize);

void cuScaleImageEx(cu_mem img/*in, out*/, size_t size, double w);

void cuAverage5x5Ex(cu_mem img/*in,out*/, const size_t xsize, const size_t ysize);

void cuMinSquareValEx(
    cu_mem img/*in,out*/,
    const size_t xsize, const size_t ysize,
    const size_t square_size, const size_t offset);

void cuMaskEx(
    ocu_channels mask/*out*/, ocu_channels mask_dc/*out*/,
    const ocu_channels &rgb, const ocu_channels &rgb2,
    const size_t xsize, const size_t ysize);

void cuCombineChannelsEx(
    cu_mem result/*out*/,
    const ocu_channels &mask,
    const ocu_channels &mask_dc,
    const size_t xsize, const size_t ysize,
    const cu_mem block_diff_dc,
    const cu_mem block_diff_ac,
    const cu_mem edge_detector_map,
    const size_t res_xsize,
    const size_t step);

void cuUpsampleSquareRootEx(cu_mem diffmap, const size_t xsize, const size_t ysize, const int step);

void cuRemoveBorderEx(cu_mem out, const cu_mem in, const size_t xsize, const size_t ysize, const int step);

void cuAddBorderEx(cu_mem out, const size_t xsize, const size_t ysize, const int step, const cu_mem in);

void cuCalculateDiffmapEx(cu_mem diffmap/*in,out*/, const size_t xsize, const size_t ysize, const int step);

#ifdef __USE_DOUBLE_AS_FLOAT__
#undef double
#endif

#endif