#pragma once
#include "CL\cl.h"
extern bool g_useOpenCL;

void clBlurEx(cl_mem image, size_t xsize, size_t ysize, double sigma, double border_ratio, cl_mem result = nullptr);

void clMinSquareVal(size_t square_size, size_t offset,
	size_t xsize, size_t ysize,
	float *values);

void clConvolution(size_t xsize, size_t ysize,
	size_t xstep,
	size_t len, size_t offset, 
	const float* multiplier,
	const float* inp,
	float border_ratio,
	float* result);

void clBlur(size_t xsize, size_t ysize, float* channel, double sigma, double border_ratio);

void clOpsinDynamicsImage(size_t xsize, size_t ysize, float* r, float* g, float* b);

void clDiffmapOpsinDynamicsImage(const float* r, const float* g, const float* b,
	float* r2, float* g2, float* b2,
	size_t xsize, size_t ysize,
	size_t step,
	float* result);