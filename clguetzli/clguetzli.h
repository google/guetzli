#pragma once
#include "CL\cl.h"
extern bool g_useOpenCL;

void clOpsinDynamicsImage(size_t xsize, size_t ysize, float* r, float* g, float& b);

void clBlurEx(cl_mem image, size_t xsize, size_t ysize, double sigma, double border_ratio);

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
