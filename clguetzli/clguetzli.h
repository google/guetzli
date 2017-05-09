#pragma once
#include "CL\cl.h"
#include "ocl.h"

extern bool g_useOpenCL;

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

void clOpsinDynamicsImage(size_t xsize, size_t ysize, float* r, float* g, float* b);

void clDiffmapOpsinDynamicsImage(const float* r, const float* g, const float* b,
	float* r2, float* g2, float* b2,
	size_t xsize, size_t ysize,
	size_t step,
	float* result);
