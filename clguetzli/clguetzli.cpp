/*
* OpenCL edition implementation of guetzli.
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*         chriskzhou@tencent.com
*/
#include "clguetzli.h"
#include <math.h>
#include <algorithm>
#include <vector>
#include "cl.hpp"

extern MATH_MODE g_mathMode = MODE_CPU;

#ifdef __USE_OPENCL__

#ifdef __USE_DOUBLE_AS_FLOAT__
#define double float
#endif

void clOpsinDynamicsImage(float *r, float *g, float *b, const size_t xsize, const size_t ysize)
{
    size_t channel_size = xsize * ysize * sizeof(float);

    ocl_args_d_t &ocl = getOcl();
    ocl_channels rgb = ocl.allocMemChannels(channel_size, r, g, b);

    clOpsinDynamicsImageEx(rgb, xsize, ysize);

    clEnqueueReadBuffer(ocl.commandQueue, rgb.r, false, 0, channel_size, r, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl.commandQueue, rgb.g, false, 0, channel_size, g, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl.commandQueue, rgb.b, false, 0, channel_size, b, 0, NULL, NULL);
    clFinish(ocl.commandQueue);

    ocl.releaseMemChannels(rgb);
}

void clDiffmapOpsinDynamicsImage(
    float* result,
    const float* r,  const float* g,  const float* b,
    const float* r2, const float* g2, const float* b2,
    const size_t xsize, const size_t ysize,
    const size_t step)
{
    size_t channel_size = xsize * ysize * sizeof(float);

    ocl_args_d_t &ocl = getOcl();
    ocl_channels xyb0 = ocl.allocMemChannels(channel_size, r, g, b);
    ocl_channels xyb1 = ocl.allocMemChannels(channel_size, r2, g2, b2);

    cl_mem mem_result = ocl.allocMem(channel_size, result);

    clDiffmapOpsinDynamicsImageEx(mem_result, xyb0, xyb1, xsize, ysize, step);

    clEnqueueReadBuffer(ocl.commandQueue, mem_result, false, 0, channel_size, result, 0, NULL, NULL);
    cl_int err = clFinish(ocl.commandQueue);

    ocl.releaseMemChannels(xyb1);
    ocl.releaseMemChannels(xyb0);

    clReleaseMemObject(mem_result);
}

void clComputeBlockZeroingOrder(
    guetzli::CoeffData *output_order_batch,
    const channel_info orig_channel[3],
    const float *orig_image_batch,
    const float *mask_scale,
    const int image_width,
    const int image_height,
    const channel_info mayout_channel[3],
    const int factor,
    const int comp_mask,
    const float BlockErrorLimit)
{
    const int block8_width = (image_width + 8 - 1) / 8;
    const int block8_height = (image_height + 8 - 1) / 8;
    const int blockf_width = (image_width + 8 * factor - 1) / (8 * factor);
    const int blockf_height = (image_height + 8 * factor - 1) / (8 * factor);

    using namespace guetzli;

    ocl_args_d_t &ocl = getOcl();

    cl_mem mem_orig_coeff[3];
    cl_mem mem_mayout_coeff[3];
    cl_mem mem_mayout_pixel[3];
    for (int c = 0; c < 3; c++)
    {
        int block_count = orig_channel[c].block_width * orig_channel[c].block_height;
        mem_orig_coeff[c] = ocl.allocMem(block_count * sizeof(::coeff_t) * kDCTBlockSize, orig_channel[c].coeff);

        block_count = mayout_channel[c].block_width * mayout_channel[c].block_height;
        mem_mayout_coeff[c] = ocl.allocMem(block_count * sizeof(::coeff_t) * kDCTBlockSize, mayout_channel[c].coeff);

        mem_mayout_pixel[c] = ocl.allocMem(image_width * image_height * sizeof(uint16_t), mayout_channel[c].pixel);
    }
    cl_mem mem_orig_image = ocl.allocMem(sizeof(float) * 3 * kDCTBlockSize * block8_width * block8_height, orig_image_batch);
    cl_mem mem_mask_scale = ocl.allocMem(sizeof(float) * 3 * block8_width * block8_height, mask_scale);

    int output_order_batch_size = sizeof(CoeffData) * 3 * kDCTBlockSize * blockf_width * blockf_height;
    cl_mem mem_output_order_batch = ocl.allocMem(output_order_batch_size, output_order_batch);

    cl_kernel kernel = ocl.kernel[KERNEL_COMPUTEBLOCKZEROINGORDER];
    clSetKernelArgEx(kernel, &mem_orig_coeff[0], &mem_orig_coeff[1], &mem_orig_coeff[2],
                        &mem_orig_image, &mem_mask_scale, 
						&blockf_width, &blockf_height,
                        &image_width, &image_height,
                        &mem_mayout_coeff[0], &mem_mayout_coeff[1], &mem_mayout_coeff[2],
                        &mem_mayout_pixel[0], &mem_mayout_pixel[1], &mem_mayout_pixel[2],
                        &mayout_channel[0], &mayout_channel[1], &mayout_channel[2],
                        &factor, 
						&comp_mask, 
						&BlockErrorLimit, 
						&mem_output_order_batch);

    size_t globalWorkSize[2] = { blockf_width, blockf_height };
    cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
    err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);

    clEnqueueReadBuffer(ocl.commandQueue, mem_output_order_batch, false, 0, output_order_batch_size, output_order_batch, 0, NULL, NULL);
    clFinish(ocl.commandQueue);

    for (int c = 0; c < 3; c++)
    {
        clReleaseMemObject(mem_orig_coeff[c]);
        clReleaseMemObject(mem_mayout_coeff[c]);
        clReleaseMemObject(mem_mayout_pixel[c]);
    }

    clReleaseMemObject(mem_orig_image);
    clReleaseMemObject(mem_mask_scale);
    clReleaseMemObject(mem_output_order_batch);
}

void clMask(
    float* mask_r,  float* mask_g,    float* mask_b,
    float* maskdc_r, float* maskdc_g, float* maskdc_b,
    const size_t xsize, const size_t ysize,
    const float* r,  const float* g,  const float* b,
    const float* r2, const float* g2, const float* b2)
{
    ocl_args_d_t &ocl = getOcl();

    size_t channel_size = xsize * ysize * sizeof(float);

    ocl_channels rgb = ocl.allocMemChannels(channel_size, r, g, b);
    ocl_channels rgb2 = ocl.allocMemChannels(channel_size, r2, g2, b2);
    ocl_channels mask = ocl.allocMemChannels(channel_size);
    ocl_channels mask_dc = ocl.allocMemChannels(channel_size);

    clMaskEx(mask, mask_dc, rgb, rgb2, xsize, ysize);

    clEnqueueReadBuffer(ocl.commandQueue, mask.r, false, 0, channel_size, mask_r, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl.commandQueue, mask.g, false, 0, channel_size, mask_g, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl.commandQueue, mask.b, false, 0, channel_size, mask_b, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl.commandQueue, mask_dc.r, false, 0, channel_size, maskdc_r, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl.commandQueue, mask_dc.g, false, 0, channel_size, maskdc_g, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl.commandQueue, mask_dc.b, false, 0, channel_size, maskdc_b, 0, NULL, NULL);
    clFinish(ocl.commandQueue);

    ocl.releaseMemChannels(rgb);
    ocl.releaseMemChannels(rgb2);
    ocl.releaseMemChannels(mask);
    ocl.releaseMemChannels(mask_dc);
}

void clDiffmapOpsinDynamicsImageEx(
    cl_mem result,
    ocl_channels xyb0,
    ocl_channels xyb1,
    const size_t xsize, const size_t ysize,
    const size_t step)
{
    const size_t res_xsize = (xsize + step - 1) / step;
    const size_t res_ysize = (ysize + step - 1) / step;

    size_t channel_size = xsize * ysize * sizeof(float);
    size_t channel_step_size = res_xsize * res_ysize * sizeof(float);

    ocl_args_d_t &ocl = getOcl();
 
    cl_mem edge_detector_map = ocl.allocMem(3 * channel_step_size);
    cl_mem block_diff_dc = ocl.allocMem(3 * channel_step_size);
    cl_mem block_diff_ac = ocl.allocMem(3 * channel_step_size);

    clMaskHighIntensityChangeEx(xyb0, xyb1, xsize, ysize);

    clEdgeDetectorMapEx(edge_detector_map, xyb0, xyb1, xsize, ysize, step);
    clBlockDiffMapEx(block_diff_dc, block_diff_ac, xyb0, xyb1, xsize, ysize, step);
    clEdgeDetectorLowFreqEx(block_diff_ac, xyb0, xyb1, xsize, ysize, step);
    {
        ocl_channels mask = ocl.allocMemChannels(channel_size);
        ocl_channels mask_dc = ocl.allocMemChannels(channel_size);
        clMaskEx(mask, mask_dc, xyb0, xyb1, xsize, ysize);
        clCombineChannelsEx(result, mask, mask_dc, xsize, ysize, block_diff_dc, block_diff_ac, edge_detector_map, res_xsize, step);

        ocl.releaseMemChannels(mask);
        ocl.releaseMemChannels(mask_dc);
    }

    clCalculateDiffmapEx(result, xsize, ysize, step);

    clReleaseMemObject(edge_detector_map);
    clReleaseMemObject(block_diff_dc);
    clReleaseMemObject(block_diff_ac);
}
void clConvolutionEx(
    cl_mem result/*out*/,
    const cl_mem inp, size_t xsize, size_t ysize,
    const cl_mem multipliers, size_t len,
    int xstep, int offset, float border_ratio)
{
	ocl_args_d_t &ocl = getOcl();

	size_t oxsize = (xsize + xstep - 1) / xstep;

	cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTION];
    clSetKernelArgEx(kernel, &result, &inp, &xsize, &multipliers, &len, &xstep, &offset, &border_ratio);

	size_t globalWorkSize[2] = { oxsize, ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);
}

void clConvolutionXEx(
    cl_mem result/*out*/,
    const cl_mem inp, size_t xsize, size_t ysize,
	const cl_mem multipliers, size_t len,
	int xstep, int offset, float border_ratio)
{
	ocl_args_d_t &ocl = getOcl();

	cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTIONX];
    clSetKernelArgEx(kernel, &result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &border_ratio);

	size_t globalWorkSize[2] = { xsize, ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);
}

void clConvolutionYEx(
    cl_mem result/*out*/,
    const cl_mem inp, size_t xsize, size_t ysize,
	const cl_mem multipliers, size_t len,
	int xstep, int offset, float border_ratio)
{
	ocl_args_d_t &ocl = getOcl();

	cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTIONY];
    clSetKernelArgEx(kernel, &result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &border_ratio);

	size_t globalWorkSize[2] = { xsize, ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);
}

void clSquareSampleEx(
    cl_mem result/*out*/,
    const cl_mem image, size_t xsize, size_t ysize,
	size_t xstep, size_t ystep)
{
	ocl_args_d_t &ocl = getOcl();

	cl_kernel kernel = ocl.kernel[KERNEL_SQUARESAMPLE];
    clSetKernelArgEx(kernel, &result, &xsize, &ysize, &image, &xstep, &ystep);

	size_t globalWorkSize[2] = { xsize, ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);
}

void clBlurEx(cl_mem image/*out, opt*/, const size_t xsize, const size_t ysize,
	const double sigma, const double border_ratio,
    cl_mem result/*out, opt*/)
{
	double m = 2.25;  // Accuracy increases when m is increased.
	const double scaler = -1.0 / (2 * sigma * sigma);
	// For m = 9.0: exp(-scaler * diff * diff) < 2^ {-52}
	const int diff = std::max<int>(1, m * fabs(sigma));
	const int expn_size = 2 * diff + 1;
	std::vector<float> expn(expn_size);
	for (int i = -diff; i <= diff; ++i) {
		expn[i + diff] = static_cast<float>(exp(scaler * i * i));
	}

	const int xstep = std::max<int>(1, int(sigma / 3));

	ocl_args_d_t &ocl = getOcl();
	cl_mem mem_expn = ocl.allocMem(sizeof(cl_float) * expn_size, expn.data());

	if (xstep > 1)
	{
        cl_mem m = ocl.allocMem(sizeof(cl_float) * xsize * ysize);
		clConvolutionXEx(m, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
		clConvolutionYEx(result ? result : image, m, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        clSquareSampleEx(result ? result : image, result ? result : image, xsize, ysize, xstep, xstep);
        clReleaseMemObject(m);
	}
	else
	{
        cl_mem m = ocl.allocMem(sizeof(cl_float) * xsize * ysize);
		clConvolutionXEx(m, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
		clConvolutionYEx(result ? result : image, m, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        clReleaseMemObject(m);
    }

	clReleaseMemObject(mem_expn);
}

void clOpsinDynamicsImageEx(ocl_channels &rgb, const size_t xsize, const size_t ysize)
{
	static const double kSigma = 1.1;

	size_t channel_size = xsize * ysize * sizeof(float);

	ocl_args_d_t &ocl = getOcl();
	ocl_channels rgb_blurred = ocl.allocMemChannels(channel_size);

    const int size = xsize * ysize;

    clBlurEx(rgb.r, xsize, ysize, kSigma, 0.0, rgb_blurred.r);
    clBlurEx(rgb.g, xsize, ysize, kSigma, 0.0, rgb_blurred.g);
    clBlurEx(rgb.b, xsize, ysize, kSigma, 0.0, rgb_blurred.b);

	cl_kernel kernel = ocl.kernel[KERNEL_OPSINDYNAMICSIMAGE];
    clSetKernelArgEx(kernel,  &rgb.r, &rgb.g, &rgb.b, &size, &rgb_blurred.r, &rgb_blurred.g, &rgb_blurred.b);

	size_t globalWorkSize[1] = { xsize * ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);

	ocl.releaseMemChannels(rgb_blurred);
}

void clMaskHighIntensityChangeEx(
    ocl_channels &xyb0/*in,out*/,
    ocl_channels &xyb1/*in,out*/,
    const size_t xsize, const size_t ysize)
{
	size_t channel_size = xsize * ysize * sizeof(float);

	ocl_args_d_t &ocl = getOcl();

	ocl_channels c0 = ocl.allocMemChannels(channel_size);
	ocl_channels c1 = ocl.allocMemChannels(channel_size);

	clEnqueueCopyBuffer(ocl.commandQueue, xyb0.r, c0.r, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb0.g, c0.g, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb0.b, c0.b, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb1.r, c1.r, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb1.g, c1.g, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb1.b, c1.b, 0, 0, channel_size, 0, NULL, NULL);
	clFinish(ocl.commandQueue);

	cl_kernel kernel = ocl.kernel[KERNEL_MASKHIGHINTENSITYCHANGE];
    clSetKernelArgEx(kernel, 
		&xyb0.r, &xyb0.g, &xyb0.b,
		&xsize, &ysize,
    	&xyb1.r, &xyb1.g, &xyb1.b,
        &c0.r, &c0.g, &c0.b,
        &c1.r, &c1.g, &c1.b);

	size_t globalWorkSize[2] = { xsize, ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);

	ocl.releaseMemChannels(c0);
	ocl.releaseMemChannels(c1);
}

void clEdgeDetectorMapEx(
    cl_mem result/*out*/,
    const ocl_channels &rgb, const ocl_channels &rgb2, 
    const size_t xsize, const size_t ysize, const size_t step)
{
	size_t channel_size = xsize * ysize * sizeof(float);
 
	ocl_args_d_t &ocl = getOcl();

	ocl_channels rgb_blured = ocl.allocMemChannels(channel_size);
	ocl_channels rgb2_blured = ocl.allocMemChannels(channel_size);

 	static const double kSigma[3] = { 1.5, 0.586, 0.4 };

	for (int i = 0; i < 3; i++)
	{
		clBlurEx(rgb.ch[i], xsize, ysize, kSigma[i], 0.0, rgb_blured.ch[i]);
		clBlurEx(rgb2.ch[i], xsize, ysize, kSigma[i], 0.0, rgb2_blured.ch[i]);
	}

    const size_t res_xsize = (xsize + step - 1) / step;
    const size_t res_ysize = (ysize + step - 1) / step;

	cl_kernel kernel = ocl.kernel[KERNEL_EDGEDETECTOR];
    clSetKernelArgEx(kernel, &result,
        &res_xsize, &res_ysize,
        &rgb_blured.r, &rgb_blured.g, &rgb_blured.b,
        &rgb2_blured.r, &rgb2_blured.g, &rgb2_blured.b,
        &xsize, &ysize, &step);

	size_t globalWorkSize[2] = { res_xsize, res_ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);

	ocl.releaseMemChannels(rgb_blured);
	ocl.releaseMemChannels(rgb2_blured);
}

void clBlockDiffMapEx(
    cl_mem block_diff_dc/*out*/, 
    cl_mem block_diff_ac/*out*/,
    const ocl_channels &rgb, const ocl_channels &rgb2,
	const size_t xsize, const size_t ysize, const size_t step)
{
	ocl_args_d_t &ocl = getOcl();


	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;
	
	cl_kernel kernel = ocl.kernel[KERNEL_BLOCKDIFFMAP];
    clSetKernelArgEx(kernel, &block_diff_dc, &block_diff_ac,
		&res_xsize, &res_ysize,
        &rgb.r, &rgb.g, &rgb.b,
        &rgb2.r, &rgb2.g, &rgb2.b,
        &xsize, &ysize, &step);


	size_t globalWorkSize[2] = { res_xsize, res_ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);
}

void clEdgeDetectorLowFreqEx(
    cl_mem block_diff_ac/*in,out*/,
    const ocl_channels &rgb, const ocl_channels &rgb2,
	const size_t xsize, const size_t ysize, const size_t step)
{
	size_t channel_size = xsize * ysize * sizeof(float);

	static const double kSigma = 14;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels rgb_blured = ocl.allocMemChannels(channel_size);
	ocl_channels rgb2_blured = ocl.allocMemChannels(channel_size);

	for (int i = 0; i < 3; i++)
	{
		clBlurEx(rgb.ch[i], xsize, ysize, kSigma, 0.0, rgb_blured.ch[i]);
		clBlurEx(rgb2.ch[i], xsize, ysize, kSigma, 0.0, rgb2_blured.ch[i]);
	}

    const size_t res_xsize = (xsize + step - 1) / step;
    const size_t res_ysize = (ysize + step - 1) / step;

	cl_kernel kernel = ocl.kernel[KERNEL_EDGEDETECTORLOWFREQ];
    clSetKernelArgEx(kernel, &block_diff_ac,
        &res_xsize, &res_ysize,
        &rgb_blured.r, &rgb_blured.g, &rgb_blured.b,
        &rgb2_blured.r, &rgb2_blured.g, &rgb2_blured.b,
        &xsize, &ysize, &step);

	size_t globalWorkSize[2] = { res_xsize, res_ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);

	ocl.releaseMemChannels(rgb_blured);
	ocl.releaseMemChannels(rgb2_blured);
}

void clDiffPrecomputeEx(
    ocl_channels &mask/*out*/,
    const ocl_channels &xyb0, const ocl_channels &xyb1, 
    const size_t xsize, const size_t ysize)
{
	ocl_args_d_t &ocl = getOcl();

	cl_kernel kernel = ocl.kernel[KERNEL_DIFFPRECOMPUTE];
    clSetKernelArgEx(kernel, &mask.x, &mask.y, &mask.b, 
							&xsize, &ysize,
                            &xyb0.x, &xyb0.y, &xyb0.b,
							&xyb1.x, &xyb1.y, &xyb1.b);

	size_t globalWorkSize[2] = { xsize, ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);
}

void clScaleImageEx(cl_mem img/*in, out*/, size_t size, double w)
{
	ocl_args_d_t &ocl = getOcl();
    float fw = w;

	cl_kernel kernel = ocl.kernel[KERNEL_SCALEIMAGE];
	clSetKernelArgEx(kernel, &img, &size, &fw);

	size_t globalWorkSize[1] = { size };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);
}

void clAverage5x5Ex(cl_mem img/*in,out*/, const size_t xsize, const size_t ysize)
{
    if (xsize < 4 || ysize < 4) {
	    // TODO: Make this work for small dimensions as well.
	    return;
    }

    ocl_args_d_t &ocl = getOcl();

    size_t len = xsize * ysize * sizeof(float);
    cl_mem img_org = ocl.allocMem(len);

    clEnqueueCopyBuffer(ocl.commandQueue, img, img_org, 0, 0, len, 0, NULL, NULL);

    cl_kernel kernel = ocl.kernel[KERNEL_AVERAGE5X5];
    clSetKernelArgEx(kernel, &img, &xsize, &ysize, &img_org);

    size_t globalWorkSize[2] = { xsize, ysize };
    cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
    err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);

    clReleaseMemObject(img_org);
}

void clMinSquareValEx(
    cl_mem img/*in,out*/, 
    const size_t xsize, const size_t ysize, 
    const size_t square_size, const size_t offset)
{
	ocl_args_d_t &ocl = getOcl();

	cl_mem result = ocl.allocMem(sizeof(cl_float) * xsize * ysize);

	cl_kernel kernel = ocl.kernel[KERNEL_MINSQUAREVAL];
    clSetKernelArgEx(kernel, &result, &xsize, &ysize, &img, &square_size, &offset);

	size_t globalWorkSize[2] = { xsize, ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clEnqueueCopyBuffer(ocl.commandQueue, result, img, 0, 0, sizeof(cl_float) * xsize * ysize, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);
    clReleaseMemObject(result);
}

static void MakeMask(double extmul, double extoff,
	double mul, double offset,
	double scaler, double *result)
{
	for (size_t i = 0; i < 512; ++i) {
		const double c = mul / ((0.01 * scaler * i) + offset);
		result[i] = 1.0 + extmul * (c + extoff);
		result[i] *= result[i];
	}
}

static const double kInternalGoodQualityThreshold = 14.921561160295326;
static const double kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

void clDoMask(ocl_channels mask/*in, out*/, ocl_channels mask_dc/*in, out*/, size_t xsize, size_t ysize)
{
	ocl_args_d_t &ocl = getOcl();

	double extmul = 0.975741017749;
	double extoff = -4.25328244168;
	double offset = 0.454909521427;
	double scaler = 0.0738288224836;
	double mul = 20.8029176447;
	static double lut_x[512];
    static bool lutx_init = false;
    if (!lutx_init)
    {
        lutx_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_x);
    }

	extmul = 0.373995618954;
	extoff = 1.5307267433;
	offset = 0.911952641929;
	scaler = 1.1731667845;
	mul = 16.2447033988;
	static double lut_y[512];
    static bool luty_init = false;
    if (!luty_init)
    {
        luty_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_y);
    }

	extmul = 0.61582234137;
	extoff = -4.25376118646;
	offset = 1.05105070921;
	scaler = 0.47434643535;
	mul = 31.1444967089;
	static double lut_b[512];
    static bool lutb_init = false;
    if (!lutb_init)
    {
        lutb_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_b);
    }

	extmul = 1.79116943438;
	extoff = -3.86797479189;
	offset = 0.670960225853;
	scaler = 0.486575865525;
	mul = 20.4563479139;
	static double lut_dcx[512];
    static bool lutdcx_init = false;
    if (!lutdcx_init)
    {
        lutdcx_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_dcx);
    }

	extmul = 0.212223514236;
	extoff = -3.65647120524;
	offset = 1.73396799447;
	scaler = 0.170392660501;
	mul = 21.6566724788;
	static double lut_dcy[512];
    static bool lutdcy_init = false;
    if (!lutdcy_init)
    {
        lutdcy_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_dcy);
    }

	extmul = 0.349376011816;
	extoff = -0.894711072781;
	offset = 0.901647926679;
	scaler = 0.380086095024;
	mul = 18.0373825149;
	static double lut_dcb[512];
    static bool lutdcb_init = false;
    if (!lutdcb_init)
    {
        lutdcb_init = true;
        MakeMask(extmul, extoff, mul, offset, scaler, lut_dcb);
    }

	size_t channel_size = 512 * sizeof(double);
	ocl_channels xyb = ocl.allocMemChannels(channel_size, lut_x, lut_y, lut_b);
    ocl_channels xyb_dc = ocl.allocMemChannels(channel_size, lut_dcx, lut_dcy, lut_dcb);

	cl_kernel kernel = ocl.kernel[KERNEL_DOMASK];
    clSetKernelArgEx(kernel, &mask.r, &mask.g, &mask.b,
        &xsize, &ysize,
        &mask_dc.r, &mask_dc.g, &mask_dc.b,
        &xyb.x, &xyb.y, &xyb.b,
        &xyb_dc.x, &xyb_dc.y, &xyb_dc.b);

	size_t globalWorkSize[2] = { xsize, ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);

	ocl.releaseMemChannels(xyb);
	ocl.releaseMemChannels(xyb_dc);
}

void clMaskEx(
    ocl_channels mask/*out*/, ocl_channels mask_dc/*out*/,
    const ocl_channels &rgb, const ocl_channels &rgb2,
	const size_t xsize, const size_t ysize)
{
    clDiffPrecomputeEx(mask, rgb, rgb2, xsize, ysize);
    for (int i = 0; i < 3; i++)
    {
        clAverage5x5Ex(mask.ch[i], xsize, ysize);
        clMinSquareValEx(mask.ch[i], xsize, ysize, 4, 0);

        static const double sigma[3] = {
            9.65781083553,
            14.2644604355,
            4.53358927369,
        };

        clBlurEx(mask.ch[i], xsize, ysize, sigma[i], 0.0);
    }

	clDoMask(mask, mask_dc, xsize, ysize);

    for (int i = 0; i < 3; i++)
    {
        clScaleImageEx(mask.ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
        clScaleImageEx(mask_dc.ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
    }
}

void clCombineChannelsEx(
    cl_mem result/*out*/,
	const ocl_channels &mask, 
	const ocl_channels &mask_dc, 
    const size_t xsize, const size_t ysize,
	const cl_mem block_diff_dc, 
	const cl_mem block_diff_ac, 
	const cl_mem edge_detector_map, 
	const size_t res_xsize,
	const size_t step)
{
	ocl_args_d_t &ocl = getOcl();

	const size_t work_xsize = ((xsize - 8 + step) + step - 1) / step;
	const size_t work_ysize = ((ysize - 8 + step) + step - 1) / step;

	cl_kernel kernel = ocl.kernel[KERNEL_COMBINECHANNELS];
    clSetKernelArgEx(kernel, &result, 
    	&mask.r, &mask.g, &mask.b,
        &mask_dc.r, &mask_dc.g, &mask_dc.b, 
        &xsize, &ysize,
        &block_diff_dc, &block_diff_ac,
        &edge_detector_map,
        &res_xsize,
        &step);

	size_t globalWorkSize[2] = { work_xsize, work_ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);
}

void clUpsampleSquareRootEx(cl_mem diffmap, const size_t xsize, const size_t ysize, const int step)
{
	ocl_args_d_t &ocl = getOcl();

    cl_mem diffmap_out = ocl.allocMem(xsize * ysize * sizeof(float));

	cl_kernel kernel = ocl.kernel[KERNEL_UPSAMPLESQUAREROOT];
    clSetKernelArgEx(kernel, &diffmap_out, &diffmap, &xsize, &ysize, &step);

	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;

	size_t globalWorkSize[2] = { res_xsize, res_ysize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clEnqueueCopyBuffer(ocl.commandQueue, diffmap_out, diffmap, 0, 0, xsize * ysize * sizeof(float), 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);

    clReleaseMemObject(diffmap_out);
}

void clRemoveBorderEx(cl_mem out, const cl_mem in, const size_t xsize, const size_t ysize, const int step)
{
	ocl_args_d_t &ocl = getOcl();

	cl_int cls = 8 - step;
	cl_int cls2 = (8 - step) / 2;

    int out_xsize = xsize - cls;
    int out_ysize = ysize - cls;

	cl_kernel kernel = ocl.kernel[KERNEL_REMOVEBORDER];
    clSetKernelArgEx(kernel, &out, &out_xsize, &out_ysize, &in, &cls, &cls2);

	size_t globalWorkSize[2] = { out_xsize, out_ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);
}

void clAddBorderEx(cl_mem out, size_t xsize, size_t ysize, int step, cl_mem in)
{
	ocl_args_d_t &ocl = getOcl();

    cl_int cls = 8 - step;
    cl_int cls2 = (8 - step) / 2;
	cl_kernel kernel = ocl.kernel[KERNEL_ADDBORDER];
    clSetKernelArgEx(kernel, &out, &xsize, &ysize, &cls, &cls2, &in);

	size_t globalWorkSize[2] = { xsize, ysize};
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    LOG_CL_RESULT(err);
	err = clFinish(ocl.commandQueue);
    LOG_CL_RESULT(err);
}

void clCalculateDiffmapEx(cl_mem diffmap/*in,out*/, const size_t xsize, const size_t ysize, const int step)
{
	clUpsampleSquareRootEx(diffmap, xsize, ysize, step);

	static const double kSigma = 8.8510880283;
	static const double mul1 = 24.8235314874;
	static const double scale = 1.0 / (1.0 + mul1);

	const int s = 8 - step;
	int s2 = (8 - step) / 2;

	ocl_args_d_t &ocl = getOcl();
	cl_mem blurred = ocl.allocMem((xsize - s) * (ysize - s) * sizeof(float));
	clRemoveBorderEx(blurred, diffmap, xsize, ysize, step);

	static const double border_ratio = 0.03027655136;
	clBlurEx(blurred, xsize - s, ysize - s, kSigma, border_ratio);

	clAddBorderEx(diffmap, xsize, ysize, step, blurred);
	clScaleImageEx(diffmap, xsize * ysize, scale);

	clReleaseMemObject(blurred);
}

#ifdef __USE_DOUBLE_AS_FLOAT__
#undef double
#endif

#endif