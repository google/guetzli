#pragma once
#include "CL\cl.h"
#include "ocl.h"

extern bool g_useOpenCL;
extern bool g_checkOpenCL;

void clOpsinDynamicsImage(size_t xsize, size_t ysize, float* r, float* g, float* b);

void clDiffmapOpsinDynamicsImage(const float* r, const float* g, const float* b,
    float* r2, float* g2, float* b2,
    size_t xsize, size_t ysize,
    size_t step,
    float* result);

void clMaskHighIntensityChangeEx(ocl_channels xyb0/*in,out*/,
	ocl_channels xyb1/*in,out*/,
	size_t xsize, size_t ysize);

void clMaskEx(ocl_channels rgb, ocl_channels rgb2,
	size_t xsize, size_t ysize,
	ocl_channels mask/*out*/, ocl_channels mask_dc/*out*/);

void clEdgeDetectorMapEx(ocl_channels rgb, ocl_channels rgb2, size_t xsize, size_t ysize, size_t step, cl_mem result/*out*/);

void clBlockDiffMapEx(ocl_channels rgb, ocl_channels rgb2,
	size_t xsize, size_t ysize, size_t step,
	cl_mem block_diff_dc/*out*/, cl_mem block_diff_ac/*out*/);

void clEdgeDetectorLowFreqEx(ocl_channels rgb, ocl_channels rgb2,
	size_t xsize, size_t ysize, size_t step,
	cl_mem block_diff_ac/*in,out*/);

void clBlurEx(cl_mem image, size_t xsize, size_t ysize, double sigma, double border_ratio, cl_mem result = nullptr);

void clOpsinDynamicsImageEx(ocl_channels rgb/*in,out*/, size_t xsize, size_t ysize);

void clCombineChannelsEx(
	ocl_channels mask,
	ocl_channels mask_dc,
	cl_mem block_diff_dc,
	cl_mem block_diff_ac,
	cl_mem edge_detector_map,
	size_t xsize, size_t ysize,
	size_t res_xsize,
	size_t step,
	cl_mem result/*out*/);

void clConvolutionEx(cl_mem inp, size_t xsize, size_t ysize,
	cl_mem multipliers, size_t len,
	int xstep, int offset, double border_ratio,
	cl_mem result/*out*/);

void clMinSquareValEx(cl_mem img/*in,out*/, size_t xsize, size_t ysize, size_t square_size, size_t offset);

void clUpsampleEx(cl_mem image, size_t xsize, size_t ysize,
	size_t xstep, size_t ystep,
	cl_mem result/*out*/);

void clCalculateDiffmapEx(cl_mem diffmap/*in,out*/, size_t xsize, size_t ysize, int step);

void clScaleImageEx(cl_mem img/*in, out*/, size_t size, double w);

void clDiffPrecomputeEx(ocl_channels xyb0, ocl_channels xyb1, size_t xsize, size_t ysize, ocl_channels mask/*out*/);

void clAverage5x5Ex(cl_mem img/*in,out*/, size_t xsize, size_t ysize);
