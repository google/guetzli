#include <CL/cl.h>
#include <math.h>
#include <assert.h>
#include "clguetzli_test.h"
#include "clguetzli.h"
#include "ocl.h"

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
	ocl_channels xyb0 = ocl.allocMemChannels(channel_size);
	ocl_channels xyb1 = ocl.allocMemChannels(channel_size);

	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.r, CL_FALSE, 0, channel_size, r, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.g, CL_FALSE, 0, channel_size, g, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.b, CL_FALSE, 0, channel_size, b, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.r, CL_FALSE, 0, channel_size, r2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.g, CL_FALSE, 0, channel_size, g2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.b, CL_FALSE, 0, channel_size, b2, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

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

	clEnqueueUnmapMemObject(ocl.commandQueue, xyb0.r, r0_r, channel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, xyb0.g, r0_g, channel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, xyb0.b, r0_b, channel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, xyb1.r, r1_r, channel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, xyb1.g, r1_g, channel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, xyb1.b, r1_b, channel_size, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(xyb0);
	ocl.releaseMemChannels(xyb1);
}

// strong to
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
	ocl_channels xyb0 = ocl.allocMemChannels(channel_size);
	ocl_channels xyb1 = ocl.allocMemChannels(channel_size);
	cl_mem edge = ocl.allocMem(edgemap_size);

	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.r, CL_FALSE, 0, channel_size, r, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.g, CL_FALSE, 0, channel_size, g, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.b, CL_FALSE, 0, channel_size, b, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.r, CL_FALSE, 0, channel_size, r2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.g, CL_FALSE, 0, channel_size, g2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.b, CL_FALSE, 0, channel_size, b2, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clEdgeDetectorMapEx(xyb0, xyb1, xsize, ysize, step, edge);

	cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, edge, true, CL_MAP_READ, 0, edgemap_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(result, r_r, res_xsize * res_ysize * 3);
	
	clEnqueueUnmapMemObject(ocl.commandQueue, edge, r_r, edgemap_size, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(xyb0);
	ocl.releaseMemChannels(xyb1);
	clReleaseMemObject(edge);
}

// strong todo
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
	ocl_channels xyb0 = ocl.allocMemChannels(channel_size);
	ocl_channels xyb1 = ocl.allocMemChannels(channel_size);
	
	cl_mem block_diff_dc = ocl.allocMem(reschannel_size);
	cl_mem block_diff_ac = ocl.allocMem(reschannel_size);

	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.r, CL_FALSE, 0, channel_size, r, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.g, CL_FALSE, 0, channel_size, g, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.b, CL_FALSE, 0, channel_size, b, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.r, CL_FALSE, 0, channel_size, r2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.g, CL_FALSE, 0, channel_size, g2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.b, CL_FALSE, 0, channel_size, b2, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clBlockDiffMapEx(xyb0, xyb1, xsize, ysize, step, block_diff_dc, block_diff_ac);

	cl_float *r_dc = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, block_diff_dc, true, CL_MAP_READ, 0, reschannel_size, 0, NULL, NULL, &err);
	cl_float *r_ac = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, block_diff_ac, true, CL_MAP_READ, 0, reschannel_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(r_dc, result_diff_dc, res_xsize * res_ysize * 3);
	FLOAT_COMPARE(r_ac, result_diff_ac, res_xsize * res_ysize * 3);

	clEnqueueUnmapMemObject(ocl.commandQueue, block_diff_dc, r_dc, reschannel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, block_diff_ac, r_ac, reschannel_size, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(xyb0);
	ocl.releaseMemChannels(xyb1);

	clReleaseMemObject(block_diff_ac);
	clReleaseMemObject(block_diff_dc);
}

// strong to
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
	ocl_channels xyb0 = ocl.allocMemChannels(channel_size);
	ocl_channels xyb1 = ocl.allocMemChannels(channel_size);

	cl_mem block_diff_ac = ocl.allocMem(reschannel_size);

	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.r, CL_FALSE, 0, channel_size, r, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.g, CL_FALSE, 0, channel_size, g, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.b, CL_FALSE, 0, channel_size, b, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.r, CL_FALSE, 0, channel_size, r2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.g, CL_FALSE, 0, channel_size, g2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.b, CL_FALSE, 0, channel_size, b2, 0, NULL, NULL);
    clEnqueueWriteBuffer(ocl.commandQueue, block_diff_ac, CL_FALSE, 0, reschannel_size, orign_ac, 0, NULL, NULL);

	err = clFinish(ocl.commandQueue);

	clEdgeDetectorLowFreqEx(xyb0, xyb1, xsize, ysize, step, block_diff_ac);

	cl_float *r_ac = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, block_diff_ac, true, CL_MAP_READ, 0, reschannel_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(r_ac, result_diff_ac, res_xsize * res_ysize * 3);

	clEnqueueUnmapMemObject(ocl.commandQueue, block_diff_ac, r_ac, reschannel_size, NULL, NULL);
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
	ocl_channels rgb = ocl.allocMemChannels(channel_size);
	ocl_channels rgb2 = ocl.allocMemChannels(channel_size);

	ocl_channels mask = ocl.allocMemChannels(channel_size);
	ocl_channels mask_dc = ocl.allocMemChannels(channel_size);

	clEnqueueWriteBuffer(ocl.commandQueue, rgb.r, CL_FALSE, 0, channel_size, r, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, rgb.g, CL_FALSE, 0, channel_size, g, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, rgb.b, CL_FALSE, 0, channel_size, b, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, rgb2.r, CL_FALSE, 0, channel_size, r2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, rgb2.g, CL_FALSE, 0, channel_size, g2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, rgb2.b, CL_FALSE, 0, channel_size, b2, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);
	
	clMaskEx(rgb, rgb2, xsize, ysize, mask/*out*/, mask_dc/*out*/);

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

	clEnqueueUnmapMemObject(ocl.commandQueue, mask.r, r0_r, channel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, mask.g, r0_g, channel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, mask.b, r0_b, channel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, mask_dc.r, r1_r, channel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, mask_dc.g, r1_g, channel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, mask_dc.b, r1_b, channel_size, NULL, NULL);
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
	float *init_result,
	float *result)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	size_t channel_size = xsize * ysize * sizeof(float);
	ocl_channels mask = ocl.allocMemChannels(channel_size);
	ocl_channels mask_dc = ocl.allocMemChannels(channel_size);
	cl_mem cl_block_diff_dc = ocl.allocMem(3 * res_xsize * res_ysize * sizeof(float));
	cl_mem cl_block_diff_ac = ocl.allocMem(3 * res_xsize * res_ysize * sizeof(float));
	cl_mem cl_edge_detector_map = ocl.allocMem(3 * res_xsize * res_ysize * sizeof(float));
	cl_mem cl_result = ocl.allocMem(res_xsize * res_ysize * sizeof(float));

	clEnqueueWriteBuffer(ocl.commandQueue, mask.x, CL_FALSE, 0, channel_size, mask_xyb_x, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, mask.y, CL_FALSE, 0, channel_size, mask_xyb_y, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, mask.b, CL_FALSE, 0, channel_size, mask_xyb_b, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, mask_dc.x, CL_FALSE, 0, channel_size, mask_xyb_dc_x, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, mask_dc.y, CL_FALSE, 0, channel_size, mask_xyb_dc_y, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, mask_dc.b, CL_FALSE, 0, channel_size, mask_xyb_dc_b, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, cl_block_diff_dc, CL_FALSE, 0, 3 * res_xsize * res_ysize * sizeof(float), block_diff_dc, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, cl_block_diff_ac, CL_FALSE, 0, 3 * res_xsize * res_ysize * sizeof(float), block_diff_ac, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, cl_edge_detector_map, CL_FALSE, 0, 3 * res_xsize * res_ysize * sizeof(float), edge_detector_map, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, cl_result, CL_FALSE, 0, res_xsize * res_ysize * sizeof(float), init_result, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clCombineChannelsEx(mask, mask_dc, cl_block_diff_dc, cl_block_diff_ac, cl_edge_detector_map, xsize, ysize, res_xsize, step, cl_result);

	cl_float *result_tmp = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, cl_result, true, CL_MAP_READ, 0, res_xsize * res_ysize * sizeof(float), 0, NULL, NULL, &err);

	FLOAT_COMPARE(result_tmp, result, res_xsize * res_ysize);

  clEnqueueUnmapMemObject(ocl.commandQueue, cl_result, result_tmp, res_xsize * res_ysize * sizeof(float), NULL, NULL);
	ocl.releaseMemChannels(mask);
	ocl.releaseMemChannels(mask_dc);
	clReleaseMemObject(cl_block_diff_dc);
	clReleaseMemObject(cl_block_diff_ac);
	clReleaseMemObject(cl_edge_detector_map);
	clReleaseMemObject(cl_result);
}

// ian todo
void tclCalculateDiffmap(const size_t xsize, const size_t ysize,
	const size_t step,
	float *diffmap, size_t org_len,
	float *diffmap_cmp)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	size_t length = xsize * ysize * sizeof(float);
	cl_mem mem_diffmap = ocl.allocMem(length);
	clEnqueueWriteBuffer(ocl.commandQueue, mem_diffmap, CL_FALSE, 0, org_len * sizeof(float), diffmap, 0, NULL, NULL);
	clCalculateDiffmapEx(mem_diffmap, xsize, ysize, step);
	//cl_float *result_tmp = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mem_diffmap, true, CL_MAP_READ, 0, length, 0, NULL, NULL, &err);
  //err = clFinish(ocl.commandQueue);
	//FLOAT_COMPARE(result_tmp, diffmap_cmp, xsize * ysize);
  //clEnqueueUnmapMemObject(ocl.commandQueue, mem_diffmap, result_tmp, length, NULL, NULL);
	clReleaseMemObject(mem_diffmap);
}

// chrisk todo
void tclBlur(float* channel, size_t xsize, size_t ysize, double sigma, double border_ratio, float* result)
{
    size_t channel_size = xsize * ysize * sizeof(float);
    cl_int err = 0;
    ocl_args_d_t &ocl = getOcl();
    cl_mem r = ocl.allocMem(channel_size);

    clEnqueueWriteBuffer(ocl.commandQueue, r, CL_FALSE, 0, channel_size, channel, 0, NULL, NULL);
    err = clFinish(ocl.commandQueue);

    clBlurEx(r, xsize, ysize, sigma, border_ratio, r);

    cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
    err = clFinish(ocl.commandQueue);

    FLOAT_COMPARE(result, r_r, xsize * ysize);

    clEnqueueUnmapMemObject(ocl.commandQueue, r, r_r, channel_size, NULL, NULL);
    err = clFinish(ocl.commandQueue);

    clReleaseMemObject(r);
}

// chrisk todo
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
	ocl.allocA(result_size);
	cl_mem r = ocl.srcA;
	cl_mem i = ocl.allocMem(inp_size);
	cl_mem m = ocl.allocMem(multipliers_size);

	clEnqueueWriteBuffer(ocl.commandQueue, i, CL_FALSE, 0, inp_size, inp, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, m, CL_FALSE, 0, multipliers_size, multipliers, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clConvolutionEx(i, xsize, ysize, m, len, xstep, offset, border_ratio, r);

	cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, r, true, CL_MAP_READ, 0, result_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(result, r_r, dxsize * ysize);

	clEnqueueUnmapMemObject(ocl.commandQueue, r, r_r, result_size, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clReleaseMemObject(i);
	clReleaseMemObject(m);
}

// chirsk todo
void tclUpsample(float* image, size_t xsize, size_t ysize,
	size_t xstep, size_t ystep,
	float* result)
{
	int dxsize = (xsize + xstep - 1) / xstep;
	int dysize = (ysize + ystep - 1) / ystep;
	size_t img_size = dxsize * dysize * sizeof(float);
	size_t result_size = xsize * ysize * sizeof(float);
	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	cl_mem img = ocl.allocMem(img_size);
	ocl.allocA(result_size);
	cl_mem r = ocl.srcA;

	clEnqueueWriteBuffer(ocl.commandQueue, img, CL_FALSE, 0, img_size, image, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clUpsampleEx(img, xsize, ysize, xstep, ystep, r);

	cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, r, true, CL_MAP_READ, 0, result_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(result, r_r, xsize * ysize);

	clEnqueueUnmapMemObject(ocl.commandQueue, r, r_r, result_size, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clReleaseMemObject(img);
}

// ian todo
void tclDiffPrecompute(
	const float *xyb0_x, const float *xyb0_y, const float *xyb0_b,
	const float *xyb1_x, const float *xyb1_y, const float *xyb1_b,
	size_t xsize, size_t ysize,
	float *mask_x, float *mask_y, float *mask_b)
{

}

// ian todo
void tclAverage5x5(int xsize, int ysize, float *diffs)
{

}

// chrisk todo
void tclMinSquareVal(float *img, size_t square_size, size_t offset,
	size_t xsize, size_t ysize,
	float *values)
{
	size_t img_size = xsize * ysize * sizeof(float);
	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	cl_mem r = ocl.allocMem(img_size);

	clEnqueueWriteBuffer(ocl.commandQueue, r, CL_FALSE, 0, img_size, img, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clMinSquareValEx(r, xsize, ysize, square_size, offset);

	cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, r, true, CL_MAP_READ, 0, img_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(values, r_r, xsize * ysize);

	clEnqueueUnmapMemObject(ocl.commandQueue, r, r_r, img_size, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clReleaseMemObject(r);
}

void tclScaleImage(double scale, float *result_org, float *result_cmp, size_t length)
{
  cl_int err = 0;
  ocl_args_d_t &ocl = getOcl();
  cl_mem mem_result_org = ocl.allocMem(length * sizeof(float));
  clEnqueueWriteBuffer(ocl.commandQueue, mem_result_org, CL_FALSE, 0, length * sizeof(float), result_org, 0, NULL, NULL);
  clScaleImageEx(mem_result_org, length, scale);

  cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mem_result_org, true, CL_MAP_READ, 0, length * sizeof(float), 0, NULL, NULL, &err);
  err = clFinish(ocl.commandQueue);

  FLOAT_COMPARE(r_r, result_cmp, length);

  clEnqueueUnmapMemObject(ocl.commandQueue, mem_result_org, r_r, length * sizeof(float), NULL, NULL);
  clReleaseMemObject(mem_result_org);
}

// strong todo
void tclOpsinDynamicsImage(float* r, float* g, float* b, size_t xsize, size_t ysize,
	float* result_r, float* result_g, float* result_b)
{
	size_t channel_size = xsize * ysize * sizeof(float);
	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels rgb = ocl.allocMemChannels(channel_size);

	clEnqueueWriteBuffer(ocl.commandQueue, rgb.r, CL_FALSE, 0, channel_size, r, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, rgb.g, CL_FALSE, 0, channel_size, g, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, rgb.b, CL_FALSE, 0, channel_size, b, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clOpsinDynamicsImageEx(rgb, xsize, ysize);

	cl_float *r_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, rgb.r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r_g = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, rgb.g, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *r_b = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, rgb.b, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(result_r, r_r, xsize * ysize);
	FLOAT_COMPARE(result_g, r_g, xsize * ysize);
	FLOAT_COMPARE(result_b, r_b, xsize * ysize);

	clEnqueueUnmapMemObject(ocl.commandQueue, rgb.r, r_r, channel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, rgb.g, r_g, channel_size, NULL, NULL);
	clEnqueueUnmapMemObject(ocl.commandQueue, rgb.b, r_b, channel_size, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(rgb);
}