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

void clMaskHighIntensityChange(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize,
	const float* result_r, const float* result_g, const float* result_b,
	const float* result_r2, const float* result_g2, const float* result_b2)
{
	if (xsize < 100 || ysize < 100) return;

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
void clEdgeDetectorMap(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize, size_t step,
	const float* result)
{
	if (xsize < 100 || ysize < 100) return;

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
void clBlockDiffMap(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize, size_t step,
	const float* result_diff_dc, const float* result_diff_ac)
{
	if (xsize < 100 || ysize < 100) return;

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
void clEdgeDetectorLowFreq(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize, size_t step,
	const float* result_diff_dc)
{
	if (xsize < 100 || ysize < 100) return;

	size_t channel_size = xsize * ysize * sizeof(float);
	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;
	const size_t reschannel_size = res_xsize * res_ysize * 3 * sizeof(float);

	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels xyb0 = ocl.allocMemChannels(channel_size);
	ocl_channels xyb1 = ocl.allocMemChannels(channel_size);

	cl_mem block_diff_dc = ocl.allocMem(reschannel_size);

	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.r, CL_FALSE, 0, channel_size, r, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.g, CL_FALSE, 0, channel_size, g, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb0.b, CL_FALSE, 0, channel_size, b, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.r, CL_FALSE, 0, channel_size, r2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.g, CL_FALSE, 0, channel_size, g2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb1.b, CL_FALSE, 0, channel_size, b2, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clEdgeDetectorLowFreqEx(xyb0, xyb1, xsize, ysize, step, block_diff_dc);

	cl_float *r_dc = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, block_diff_dc, true, CL_MAP_READ, 0, reschannel_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	FLOAT_COMPARE(r_dc, result_diff_dc, res_xsize * res_ysize * 3);

	clEnqueueUnmapMemObject(ocl.commandQueue, block_diff_dc, r_dc, reschannel_size, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	ocl.releaseMemChannels(xyb0);
	ocl.releaseMemChannels(xyb1);

	clReleaseMemObject(block_diff_dc);
}

void clMask(const float* r, const float* g, const float* b,
	const float* r2, const float* g2, const float* b2,
	size_t xsize, size_t ysize,
	const float* mask_r, const float* mask_g, const float* mask_b,
	const float* maskdc_r, const float* maskdc_g, const float* maskdc_b)
{
	if (xsize < 100 || ysize < 100) return;

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

// ian todo
void clCombineChannels(void)
{

}

// ian todo
void clCalculateDiffmapEx(void)
{

}

// strong todo
void clBlur(void)
{

}

// strong todo
void clConvolution(void)
{

}

// strong todo
void clUpsample(void)
{

}

// ian todo
void clDiffPrecompute(void)
{

}

// ian todo
void clAverage5x5(void)
{

}

// strong todo
void clMinSquareVal(void)
{

}

// ian todo
void clScaleImage(void)
{

}
