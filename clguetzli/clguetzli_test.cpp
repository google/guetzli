/*
* OpenCL test cases
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*         chriskzhou@tencent.com
*/
#ifdef __USE_OPENCL__

#include <CL/cl.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include "clguetzli_test.h"
#include "clguetzli.h"
#include "ocl.h"
#include "ocu.h"

#define FLOAT_COMPARE(a, b, c)  floatCompare((a), (b), (c), __FUNCTION__, __LINE__ )

int floatCompare(const float* a, const float* b, size_t size, const char* szFunc, int line)
{
	int count = 0;
	for (int i = 0; i < size; i++)
	{
		if (fabs(a[i] - b[i]) > 0.001)
		{
			count++;
		}
	}
	if (count > 0)
	{
		LogError("CHK %s(%d) %d:%d\r\n", szFunc, line, count, size);
	}
	return count;
}

void tclMaskHighIntensityChange(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize,
	const float* result_r, const float* result_g, const float* result_b,
	const float* result_r2, const float* result_g2, const float* result_b2)
{
	size_t channel_size = xsize * ysize * sizeof(float);
	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels xyb0 = ocl.allocMemChannels(channel_size, r, g, b);
	ocl_channels xyb1 = ocl.allocMemChannels(channel_size, r2, g2, b2);

	clMaskHighIntensityChangeEx(xyb0, xyb1, xsize, ysize);

	cl_float *r0_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, xyb0.r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r0_g = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, xyb0.g, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r0_b = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, xyb0.b, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r1_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, xyb1.r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r1_g = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, xyb1.g, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r1_b = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, xyb1.b, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(result_r, r0_r, xsize * ysize);
	FLOAT_COMPARE(result_g, r0_g, xsize * ysize);
	FLOAT_COMPARE(result_b, r0_b, xsize * ysize);
	FLOAT_COMPARE(result_r2, r1_r, xsize * ysize);
	FLOAT_COMPARE(result_g2, r1_g, xsize * ysize);
	FLOAT_COMPARE(result_b2, r1_b, xsize * ysize);

	clEnqueueUnmapMemObject(ocl.commandQueue, xyb0.r, r0_r, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, xyb0.g, r0_g, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, xyb0.b, r0_b, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, xyb1.r, r1_r, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, xyb1.g, r1_g, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, xyb1.b, r1_b, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(xyb0);
	ocl.releaseMemChannels(xyb1);
}

void tclEdgeDetectorMap(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize, size_t step,
	const float* result)
{
	size_t channel_size = xsize * ysize * sizeof(float);
	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;
	const size_t edgemap_size = res_xsize * res_ysize * 3 * sizeof(float);

	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels xyb0 = ocl.allocMemChannels(channel_size, r, g, b);
	ocl_channels xyb1 = ocl.allocMemChannels(channel_size, r2, g2, b2);
	cl_mem edge = ocl.allocMem(edgemap_size);

	clEdgeDetectorMapEx(edge, xyb0, xyb1, xsize, ysize, step);

	cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, edge, true, CL_MAP_READ, 0, edgemap_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(result, r_r, res_xsize * res_ysize * 3);
	
	clEnqueueUnmapMemObject(ocl.commandQueue, edge, r_r, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(xyb0);
	ocl.releaseMemChannels(xyb1);
	clReleaseMemObject(edge);
}

void tclBlockDiffMap(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize, size_t step,
	const float* result_diff_dc, const float* result_diff_ac)
{
	size_t channel_size = xsize * ysize * sizeof(float);
	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;
	const size_t reschannel_size = res_xsize * res_ysize * 3 * sizeof(float);

	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels xyb0 = ocl.allocMemChannels(channel_size, r, g, b);
	ocl_channels xyb1 = ocl.allocMemChannels(channel_size, r2, g2, b2);
	
	cl_mem block_diff_dc = ocl.allocMem(reschannel_size);
	cl_mem block_diff_ac = ocl.allocMem(reschannel_size);

	clBlockDiffMapEx(block_diff_dc, block_diff_ac, xyb0, xyb1, xsize, ysize, step);

	cl_float *r_dc = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, block_diff_dc, true, CL_MAP_READ, 0, reschannel_size, 0, NULL, NULL, &err);
	cl_float *r_ac = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, block_diff_ac, true, CL_MAP_READ, 0, reschannel_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(r_dc, result_diff_dc, res_xsize * res_ysize * 3);
	FLOAT_COMPARE(r_ac, result_diff_ac, res_xsize * res_ysize * 3);

	clEnqueueUnmapMemObject(ocl.commandQueue, block_diff_dc, r_dc, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, block_diff_ac, r_ac, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(xyb0);
	ocl.releaseMemChannels(xyb1);

	clReleaseMemObject(block_diff_ac);
	clReleaseMemObject(block_diff_dc);
}

void tclEdgeDetectorLowFreq(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize, size_t step,
    const float* orign_ac,
	const float* result_diff_ac)
{
	size_t channel_size = xsize * ysize * sizeof(float);
	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;
	const size_t reschannel_size = res_xsize * res_ysize * 3 * sizeof(float);

	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels xyb0 = ocl.allocMemChannels(channel_size, r, g, b);
	ocl_channels xyb1 = ocl.allocMemChannels(channel_size, r2, g2, b2);

	cl_mem block_diff_ac = ocl.allocMem(reschannel_size, orign_ac);

	clEdgeDetectorLowFreqEx(block_diff_ac, xyb0, xyb1, xsize, ysize, step);

	cl_float *r_ac = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, block_diff_ac, true, CL_MAP_READ, 0, reschannel_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(r_ac, result_diff_ac, res_xsize * res_ysize * 3);

	clEnqueueUnmapMemObject(ocl.commandQueue, block_diff_ac, r_ac, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(xyb0);
	ocl.releaseMemChannels(xyb1);

	clReleaseMemObject(block_diff_ac);
}

void tclMask(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize,
	const float* mask_r, const float* mask_g, const float* mask_b,
	const float* maskdc_r, const float* maskdc_g, const float* maskdc_b)
{
	size_t channel_size = xsize * ysize * sizeof(float);
	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels rgb = ocl.allocMemChannels(channel_size, r, g, b);
	ocl_channels rgb2 = ocl.allocMemChannels(channel_size, r2, g2, b2);

	ocl_channels mask = ocl.allocMemChannels(channel_size);
	ocl_channels mask_dc = ocl.allocMemChannels(channel_size);
    	
	clMaskEx(mask/*out*/, mask_dc/*out*/, rgb, rgb2, xsize, ysize);

	cl_float *r0_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mask.r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r0_g = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mask.g, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r0_b = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mask.b, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r1_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mask_dc.r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r1_g = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mask_dc.g, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r1_b = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mask_dc.b, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(mask_r, r0_r, xsize * ysize);
	FLOAT_COMPARE(mask_g, r0_g, xsize * ysize);
	FLOAT_COMPARE(mask_b, r0_b, xsize * ysize);
	FLOAT_COMPARE(maskdc_r, r1_r, xsize * ysize);
	FLOAT_COMPARE(maskdc_g, r1_g, xsize * ysize);
	FLOAT_COMPARE(maskdc_b, r1_b, xsize * ysize);

	clEnqueueUnmapMemObject(ocl.commandQueue, mask.r, r0_r, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, mask.g, r0_g, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, mask.b, r0_b, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, mask_dc.r, r1_r, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, mask_dc.g, r1_g, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, mask_dc.b, r1_b, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(rgb);
	ocl.releaseMemChannels(rgb2);
	ocl.releaseMemChannels(mask);
	ocl.releaseMemChannels(mask_dc);
}

void tclCombineChannels(const float *mask_xyb_x, const float *mask_xyb_y, const float *mask_xyb_b,
	const float *mask_xyb_dc_x, const float *mask_xyb_dc_y, const float *mask_xyb_dc_b,
	const float *block_diff_dc,	const float *block_diff_ac,
	const float *edge_detector_map,
	size_t xsize, size_t ysize,
	size_t res_xsize, size_t res_ysize,
	size_t step,
	const float *init_result,
	const float *result)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	size_t channel_size = xsize * ysize * sizeof(float);
    size_t res_channel_size = res_xsize * res_ysize * sizeof(float);
	ocl_channels mask = ocl.allocMemChannels(channel_size, mask_xyb_x, mask_xyb_y, mask_xyb_b);
	ocl_channels mask_dc = ocl.allocMemChannels(channel_size, mask_xyb_dc_x, mask_xyb_dc_y, mask_xyb_dc_b);
	cl_mem cl_block_diff_dc = ocl.allocMem(3 * res_channel_size, block_diff_dc);
	cl_mem cl_block_diff_ac = ocl.allocMem(3 * res_channel_size, block_diff_ac);
	cl_mem cl_edge_detector_map = ocl.allocMem(3 * res_channel_size, edge_detector_map);
	cl_mem cl_result = ocl.allocMem(res_channel_size, init_result);

	clCombineChannelsEx(cl_result, mask, mask_dc, xsize, ysize, cl_block_diff_dc, cl_block_diff_ac, cl_edge_detector_map, res_xsize, step);

	cl_float *result_tmp = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, cl_result, true, CL_MAP_READ, 0, res_xsize * res_ysize * sizeof(float), 0, NULL, NULL, &err);

	FLOAT_COMPARE(result_tmp, result, res_xsize * res_ysize);

    clEnqueueUnmapMemObject(ocl.commandQueue, cl_result, result_tmp, 0, NULL, NULL);
	ocl.releaseMemChannels(mask);
	ocl.releaseMemChannels(mask_dc);
	clReleaseMemObject(cl_block_diff_dc);
	clReleaseMemObject(cl_block_diff_ac);
	clReleaseMemObject(cl_edge_detector_map);
	clReleaseMemObject(cl_result);
}

void tclCalculateDiffmap(const size_t xsize, const size_t ysize,
	const size_t step,
	const float *diffmap, size_t org_len,
	const float *diffmap_cmp)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	size_t length = xsize * ysize * sizeof(float);
	cl_mem mem_diffmap = ocl.allocMem(length);
	clEnqueueWriteBuffer(ocl.commandQueue, mem_diffmap, CL_FALSE, 0, org_len * sizeof(float), diffmap, 0, NULL, NULL);
	clCalculateDiffmapEx(mem_diffmap, xsize, ysize, step);
	cl_float *result_tmp = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mem_diffmap, true, CL_MAP_READ, 0, length, 0, NULL, NULL, &err);
    err = clFinish(ocl.commandQueue);
	FLOAT_COMPARE(result_tmp, diffmap_cmp, xsize * ysize);
    clEnqueueUnmapMemObject(ocl.commandQueue, mem_diffmap, result_tmp, 0, NULL, NULL);
	clReleaseMemObject(mem_diffmap);
}

void tclBlur(const float* channel, size_t xsize, size_t ysize, double sigma, double border_ratio, const float* result)
{
    size_t channel_size = xsize * ysize * sizeof(float);
    cl_int err = 0;
    ocl_args_d_t &ocl = getOcl();
    cl_mem r = ocl.allocMem(channel_size, channel);

    clBlurEx(r, xsize, ysize, sigma, border_ratio, r);

    cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
    err = clFinish(ocl.commandQueue);

    FLOAT_COMPARE(result, r_r, xsize * ysize);

    clEnqueueUnmapMemObject(ocl.commandQueue, r, r_r, 0, NULL, NULL);
    err = clFinish(ocl.commandQueue);

    clReleaseMemObject(r);
}

void tclConvolution(size_t xsize, size_t ysize,
	size_t xstep,
	size_t len, size_t offset,
	const float* multipliers,
	const float* inp,
	float border_ratio,
	float* result)
{
	int dxsize = (xsize + xstep - 1) / xstep;
	size_t result_size = dxsize * ysize * sizeof(float);
	size_t inp_size = xsize * ysize * sizeof(float);
	size_t multipliers_size = len * sizeof(float);
	cl_int err = 0;
    ocl_args_d_t &ocl = getOcl();
    cl_mem r = ocl.allocMem(result_size);
	cl_mem i = ocl.allocMem(inp_size, inp);
	cl_mem m = ocl.allocMem(multipliers_size, multipliers);

	clConvolutionEx(r, i, xsize, ysize, m, len, xstep, offset, border_ratio);

	cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, r, true, CL_MAP_READ, 0, result_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(result, r_r, dxsize * ysize);

	clEnqueueUnmapMemObject(ocl.commandQueue, r, r_r, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

    clReleaseMemObject(r);
	clReleaseMemObject(i);
	clReleaseMemObject(m);
}

void tclDiffPrecompute(
  const std::vector<std::vector<float> > &xyb0,
  const std::vector<std::vector<float> > &xyb1,
  size_t xsize, size_t ysize,
  const std::vector<std::vector<float> > *mask_cmp)
{
  cl_int err = 0;
  ocl_args_d_t &ocl = getOcl();
  size_t channel_size = xsize * ysize * sizeof(float);
  ocl_channels cl_xyb0 = ocl.allocMemChannels(channel_size, xyb0[0].data(), xyb0[1].data(), xyb0[2].data());
  ocl_channels cl_xyb1 = ocl.allocMemChannels(channel_size, xyb1[0].data(), xyb1[1].data(), xyb1[2].data());
  ocl_channels cl_mask = ocl.allocMemChannels(channel_size);

  clDiffPrecomputeEx(cl_mask, cl_xyb0, cl_xyb1, xsize, ysize);

  cl_float *r_x = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, cl_mask.x, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
  cl_float *r_y = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, cl_mask.y, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
  cl_float *r_b = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, cl_mask.b, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
  err = clFinish(ocl.commandQueue);

  FLOAT_COMPARE(r_x, (*mask_cmp)[0].data(), xsize * ysize);
  FLOAT_COMPARE(r_y, (*mask_cmp)[1].data(), xsize * ysize);
  FLOAT_COMPARE(r_b, (*mask_cmp)[2].data(), xsize * ysize);

  clEnqueueUnmapMemObject(ocl.commandQueue, cl_mask.x, r_x, 0, NULL, NULL);
  clEnqueueUnmapMemObject(ocl.commandQueue, cl_mask.y, r_y, 0, NULL, NULL);
  clEnqueueUnmapMemObject(ocl.commandQueue, cl_mask.b, r_b, 0, NULL, NULL);
  ocl.releaseMemChannels(cl_xyb0);
  ocl.releaseMemChannels(cl_xyb1);
  ocl.releaseMemChannels(cl_mask);
}

void tclAverage5x5(int xsize, int ysize, const std::vector<float> &diffs_org, const std::vector<float> &diffs_cmp)
{
  cl_int err = 0;
  ocl_args_d_t &ocl = getOcl();
  cl_mem mem_diff = ocl.allocMem(xsize * ysize * sizeof(float), diffs_org.data());

  clAverage5x5Ex(mem_diff, xsize, ysize);
  cl_float *r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mem_diff, true, CL_MAP_READ, 0, xsize * ysize * sizeof(float), 0, NULL, NULL, &err);
  err = clFinish(ocl.commandQueue);
  FLOAT_COMPARE(r, diffs_cmp.data(), xsize * ysize);

  clEnqueueUnmapMemObject(ocl.commandQueue, mem_diff, r, 0, NULL, NULL);
  clReleaseMemObject(mem_diff);
}

void tclMinSquareVal(const float *img, size_t square_size, size_t offset,
	size_t xsize, size_t ysize,
	const float *result)
{
	size_t img_size = xsize * ysize * sizeof(float);
	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	cl_mem r = ocl.allocMem(img_size, img);

	clMinSquareValEx(r, xsize, ysize, square_size, offset);

	cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, r, true, CL_MAP_READ, 0, img_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(result, r_r, xsize * ysize);

	clEnqueueUnmapMemObject(ocl.commandQueue, r, r_r, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clReleaseMemObject(r);
}

void tclScaleImage(double scale, const float *result_org, const float *result_cmp, size_t length)
{
    cl_int err = 0;
    ocl_args_d_t &ocl = getOcl();
    cl_mem mem_result_org = ocl.allocMem(length * sizeof(float), result_org);

    clScaleImageEx(mem_result_org, length, scale);

    cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mem_result_org, true, CL_MAP_READ, 0, length * sizeof(float), 0, NULL, NULL, &err);
    err = clFinish(ocl.commandQueue);

    FLOAT_COMPARE(r_r, result_cmp, length);

    clEnqueueUnmapMemObject(ocl.commandQueue, mem_result_org, r_r, 0, NULL, NULL);
    clReleaseMemObject(mem_result_org);
}

void tclOpsinDynamicsImage(const float* r, const float* g, const float* b, size_t xsize, size_t ysize,
	const float* result_r, const float* result_g, const float* result_b)
{
	size_t channel_size = xsize * ysize * sizeof(float);
	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels rgb = ocl.allocMemChannels(channel_size, r, g, b);

	clOpsinDynamicsImageEx(rgb, xsize, ysize);

	cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, rgb.r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r_g = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, rgb.g, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r_b = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, rgb.b, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(result_r, r_r, xsize * ysize);
	FLOAT_COMPARE(result_g, r_g, xsize * ysize);
	FLOAT_COMPARE(result_b, r_b, xsize * ysize);

	clEnqueueUnmapMemObject(ocl.commandQueue, rgb.r, r_r, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, rgb.g, r_g, 0, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, rgb.b, r_b, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(rgb);
}

#endif