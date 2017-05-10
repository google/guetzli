#pragma once
#include "ocl.h"

ocl_args_d_t& getOcl(void);

void tclMaskHighIntensityChange(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize,
	const float* result_r, const float* result_g, const float* result_b,
	const float* result_r2, const float* result_g2, const float* result_b2);

void tclBlur(float* channel, size_t xsize, size_t ysize, double sigma, double border_ratio, float* result);

void tclEdgeDetectorMap(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize, size_t step,
	const float* result);

void tclBlockDiffMap(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize, size_t step,
	const float* result_diff_dc, const float* result_diff_ac);

void tclEdgeDetectorLowFreq(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize, size_t step,
    const float* orign_ac,
	const float* result_diff_dc);

void tclMask(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize,
	const float* mask_r, const float* mask_g, const float* mask_b,
	const float* maskdc_r, const float* maskdc_g, const float* maskdc_b);

void tclCombineChannels(const float *mask_xyb_x, const float *mask_xyb_y, const float *mask_xyb_b,
	const float *mask_xyb_dc_x, const float *mask_xyb_dc_y, const float *mask_xyb_dc_b,
	const float *block_diff_dc, const float *block_diff_ac,
	const float *edge_detector_map,
	size_t xsize, size_t ysize,
	size_t res_xsize, size_t res_ysize,
	size_t step,
	float *result);

void tclCalculateDiffmap(const size_t xsize, const size_t ysize,
	const size_t step,
	float *diffmap);

void tclBlur(size_t xsize, size_t ysize, float* channel, double sigma,
	double border_ratio);

void tclDiffPrecompute(
	const float *xyb0_x, const float *xyb0_y, const float *xyb0_b,
	const float *xyb1_x, const float *xyb1_y, const float *xyb1_b,
	size_t xsize, size_t ysize,
	float *mask_x, float *mask_y, float *mask_b);

void tclAverage5x5(int xsize, int ysize, float *diffs);

void tclScaleImage(double scale, float *result);

void tclOpsinDynamicsImage(float* r, float* g, float* b, size_t xsize, size_t ysize,
	float* result_r, float* result_g, float* result_b);
