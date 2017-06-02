#pragma once
#include "guetzli/processor.h"
#include "clguetzli.cl.h"

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

void cuConvolutionXEx(
    CUdeviceptr result/*out*/,
    const CUdeviceptr inp, size_t xsize, size_t ysize,
    const CUdeviceptr multipliers, size_t len,
    int xstep, int offset, double border_ratio);

void cuConvolutionYEx(
    CUdeviceptr result/*out*/,
    const CUdeviceptr inp, size_t xsize, size_t ysize,
    const CUdeviceptr multipliers, size_t len,
    int xstep, int offset, double border_ratio);

void cuSquareSampleEx(
    CUdeviceptr result/*out*/,
    const CUdeviceptr image, size_t xsize, size_t ysize,
    size_t xstep, size_t ystep);

void cuBlurEx(CUdeviceptr image/*out, opt*/, const size_t xsize, const size_t ysize,
    const double sigma, const double border_ratio,
    CUdeviceptr result = NULL/*out, opt*/);

void cuOpsinDynamicsImageEx(ocu_channels &rgb, const size_t xsize, const size_t ysize);

void cuMaskHighIntensityChangeEx(
    ocu_channels &xyb0/*in,out*/,
    ocu_channels &xyb1/*in,out*/,
    const size_t xsize, const size_t ysize);

void cuEdgeDetectorMapEx(
    CUdeviceptr result/*out*/,
    const ocu_channels &rgb, const ocu_channels &rgb2,
    const size_t xsize, const size_t ysize, const size_t step);

void cuBlockDiffMapEx(
    CUdeviceptr block_diff_dc/*out*/,
    CUdeviceptr block_diff_ac/*out*/,
    const ocu_channels &rgb, const ocu_channels &rgb2,
    const size_t xsize, const size_t ysize, const size_t step);

void cuEdgeDetectorLowFreqEx(
    CUdeviceptr block_diff_ac/*in,out*/,
    const ocu_channels &rgb, const ocu_channels &rgb2,
    const size_t xsize, const size_t ysize, const size_t step);

void cuDiffPrecomputeEx(
    ocu_channels &mask/*out*/,
    const ocu_channels &xyb0, const ocu_channels &xyb1,
    const size_t xsize, const size_t ysize);

void cuScaleImageEx(CUdeviceptr img/*in, out*/, size_t size, double w);

void cuAverage5x5Ex(CUdeviceptr img/*in,out*/, const size_t xsize, const size_t ysize);

void cuMinSquareValEx(
    CUdeviceptr img/*in,out*/,
    const size_t xsize, const size_t ysize,
    const size_t square_size, const size_t offset);

void cuMaskEx(
    ocu_channels mask/*out*/, ocu_channels mask_dc/*out*/,
    const ocu_channels &rgb, const ocu_channels &rgb2,
    const size_t xsize, const size_t ysize);

void cuCombineChannelsEx(
    CUdeviceptr result/*out*/,
    const ocu_channels &mask,
    const ocu_channels &mask_dc,
    const size_t xsize, const size_t ysize,
    const CUdeviceptr block_diff_dc,
    const CUdeviceptr block_diff_ac,
    const CUdeviceptr edge_detector_map,
    const size_t res_xsize,
    const size_t step);

void cuUpsampleSquareRootEx(CUdeviceptr diffmap, const size_t xsize, const size_t ysize, const int step);

void cuRemoveBorderEx(CUdeviceptr out, const CUdeviceptr in, const size_t xsize, const size_t ysize, const int step);

void cuAddBorderEx(CUdeviceptr out, const size_t xsize, const size_t ysize, const int step, const CUdeviceptr in);

void cuCalculateDiffmapEx(CUdeviceptr diffmap/*in,out*/, const size_t xsize, const size_t ysize, const int step);
