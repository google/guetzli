/*
* OpenCL test cases
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*         chriskzhou@tencent.com
*/
#pragma once
#include "ocl.h"

void tclMaskHighIntensityChange(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize,
	const float* result_r, const float* result_g, const float* result_b,
	const float* result_r2, const float* result_g2, const float* result_b2);

void tclBlur(const float* channel, size_t xsize, size_t ysize, double sigma, double border_ratio, const float* result);

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
	const float *init_result,
	const float *result);

void tclCalculateDiffmap(const size_t xsize, const size_t ysize,
	const size_t step,
	const float *diffmap, size_t org_len,
	const float *diffmap_cmp);

void tclConvolution(size_t xsize, size_t ysize,
	size_t xstep,
	size_t len, size_t offset,
	const float* multipliers,
	const float* inp,
	float border_ratio,
	float* result);

void tclDiffPrecompute(
  const std::vector<std::vector<float> > &xyb0,
  const std::vector<std::vector<float> > &xyb1,
  size_t xsize, size_t ysize,
  const std::vector<std::vector<float> > *mask_cmp);

void tclAverage5x5(int xsize, int ysize, const std::vector<float> &diffs_org, const std::vector<float> &diffs_cmp);

void tclScaleImage(double scale, const float *result_org, const float *result_cmp, size_t length);

void tclOpsinDynamicsImage(const float* r, const float* g, const float* b, size_t xsize, size_t ysize,
	const float* result_r, const float* result_g, const float* result_b);

void tclMinSquareVal(const float *img, size_t square_size, size_t offset,
	size_t xsize, size_t ysize,
	const float *result);
