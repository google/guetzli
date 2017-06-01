#include <math.h>
#include <algorithm>
#include <vector>
#include "clguetzli.h"
#include "ocu.h"

extern bool g_useOpenCL = false;
extern bool g_useCuda = false;
extern bool g_checkOpenCL = false;

ocl_args_d_t& getOcl(void)
{
	static bool bInit = false;
	static ocl_args_d_t ocl;

	if (bInit == true) return ocl;

	bInit = true;
	cl_int err = SetupOpenCL(&ocl, CL_DEVICE_TYPE_GPU);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));
	}

	char* source = nullptr;
	size_t src_size = 0;
	ReadSourceFromFile("clguetzli/clguetzli.cl", &source, &src_size);

	ocl.program = clCreateProgramWithSource(ocl.context, 1, (const char**)&source, &src_size, &err);

	delete[] source;

	err = clBuildProgram(ocl.program, 1, &ocl.device, "", NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));

        if (err == CL_BUILD_PROGRAM_FAILURE)
        {
            size_t log_size = 0;
            clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

            std::vector<char> build_log(log_size);
            clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], NULL);

            LogError("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
        }
	}
    ocl.kernel[KERNEL_CONVOLUTION] = clCreateKernel(ocl.program, "clConvolutionEx", &err);
    ocl.kernel[KERNEL_CONVOLUTIONX] = clCreateKernel(ocl.program, "clConvolutionXEx", &err);
    ocl.kernel[KERNEL_CONVOLUTIONY] = clCreateKernel(ocl.program, "clConvolutionYEx", &err);
    ocl.kernel[KERNEL_SQUARESAMPLE] = clCreateKernel(ocl.program, "clSquareSampleEx", &err);
	ocl.kernel[KERNEL_OPSINDYNAMICSIMAGE] = clCreateKernel(ocl.program, "clOpsinDynamicsImageEx", &err);
    ocl.kernel[KERNEL_MASKHIGHINTENSITYCHANGE] = clCreateKernel(ocl.program, "clMaskHighIntensityChangeEx", &err);
    ocl.kernel[KERNEL_EDGEDETECTOR] = clCreateKernel(ocl.program, "clEdgeDetectorMapEx", &err);
    ocl.kernel[KERNEL_BLOCKDIFFMAP] = clCreateKernel(ocl.program, "clBlockDiffMapEx", &err);
    ocl.kernel[KERNEL_EDGEDETECTORLOWFREQ] = clCreateKernel(ocl.program, "clEdgeDetectorLowFreqEx", &err);
    ocl.kernel[KERNEL_DIFFPRECOMPUTE] = clCreateKernel(ocl.program, "clDiffPrecomputeEx", &err);
    ocl.kernel[KERNEL_SCALEIMAGE] = clCreateKernel(ocl.program, "clScaleImageEx", &err);
    ocl.kernel[KERNEL_AVERAGE5X5] = clCreateKernel(ocl.program, "clAverage5x5Ex", &err);
    ocl.kernel[KERNEL_MINSQUAREVAL] = clCreateKernel(ocl.program, "clMinSquareValEx", &err);
    ocl.kernel[KERNEL_DOMASK] = clCreateKernel(ocl.program, "clDoMaskEx", &err);
    ocl.kernel[KERNEL_COMBINECHANNELS] = clCreateKernel(ocl.program, "clCombineChannelsEx", &err);
    ocl.kernel[KERNEL_UPSAMPLESQUAREROOT] = clCreateKernel(ocl.program, "clUpsampleSquareRootEx", &err);
    ocl.kernel[KERNEL_REMOVEBORDER] = clCreateKernel(ocl.program, "clRemoveBorderEx", &err);
    ocl.kernel[KERNEL_ADDBORDER] = clCreateKernel(ocl.program, "clAddBorderEx", &err);
    ocl.kernel[KERNEL_COMPUTEBLOCKZEROINGORDER] = clCreateKernel(ocl.program, "clComputeBlockZeroingOrderEx", &err);

	return ocl;
}

void clOpsinDynamicsImage(float *r, float *g, float *b, const size_t xsize, const size_t ysize)
{
    cl_int channel_size = xsize * ysize * sizeof(float);

    cl_int err = 0;
    ocl_args_d_t &ocl = getOcl();
    ocl_channels rgb = ocl.allocMemChannels(channel_size, r, g, b);

    clOpsinDynamicsImageEx(rgb, xsize, ysize);

    cl_float *result_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, rgb.r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
    cl_float *result_g = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, rgb.g, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
    cl_float *result_b = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, rgb.b, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);

    err = clFinish(ocl.commandQueue);

    memcpy(r, result_r, channel_size);
    memcpy(g, result_g, channel_size);
    memcpy(b, result_b, channel_size);

    clEnqueueUnmapMemObject(ocl.commandQueue, rgb.r, result_r, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl.commandQueue, rgb.g, result_g, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl.commandQueue, rgb.b, result_b, 0, NULL, NULL);
    clFinish(ocl.commandQueue);

    ocl.releaseMemChannels(rgb);
}

void clDiffmapOpsinDynamicsImage(
    float* result,
    const float* r,  const float* g,  const float* b,
    const float* r2, const float* g2, const float* b2,
    size_t xsize, size_t ysize,
    size_t step)
{

    const size_t res_xsize = (xsize + step - 1) / step;
    const size_t res_ysize = (ysize + step - 1) / step;

    cl_int channel_size = xsize * ysize * sizeof(float);
    cl_int channel_step_size = res_xsize * res_ysize * sizeof(float);

    cl_int err = 0;
    ocl_args_d_t &ocl = getOcl();
    ocl_channels xyb0 = ocl.allocMemChannels(channel_size, r, g, b);
    ocl_channels xyb1 = ocl.allocMemChannels(channel_size, r2, g2, b2);

    cl_mem mem_result = ocl.allocMem(channel_size, result);

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
        clCombineChannelsEx(mem_result, mask, mask_dc, xsize, ysize, block_diff_dc, block_diff_ac, edge_detector_map, res_xsize, step);

        ocl.releaseMemChannels(mask);
        ocl.releaseMemChannels(mask_dc);
    }

    clCalculateDiffmapEx(mem_result, xsize, ysize, step);

    cl_float *result_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mem_result, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
    err = clFinish(ocl.commandQueue);
    memcpy(result, result_r, channel_size);

    clEnqueueUnmapMemObject(ocl.commandQueue, mem_result, result_r, 0, NULL, NULL);
    clFinish(ocl.commandQueue);

    ocl.releaseMemChannels(xyb1);
    ocl.releaseMemChannels(xyb0);

    clReleaseMemObject(edge_detector_map);
    clReleaseMemObject(block_diff_dc);
    clReleaseMemObject(block_diff_ac);

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

    cl_int err = 0;
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
    cl_mem mem_output_order_batch = ocl.allocMem(output_order_batch_size);
    cl_float clBlockErrorLimit = BlockErrorLimit;
    cl_int clWidth = image_width;
    cl_int clHeight = image_height;
    cl_int clFactor = factor;
    cl_int clMask = comp_mask;

	clEnqueueWriteBuffer(ocl.commandQueue, mem_output_order_batch, CL_FALSE, 0, output_order_batch_size, output_order_batch, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

    cl_kernel kernel = ocl.kernel[KERNEL_COMPUTEBLOCKZEROINGORDER];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_orig_coeff[0]);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_orig_coeff[1]);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mem_orig_coeff[2]);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&mem_orig_image);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&mem_mask_scale);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &clWidth);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &clHeight);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&mem_mayout_coeff[0]);
    clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&mem_mayout_coeff[1]);
    clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&mem_mayout_coeff[2]);
    clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&mem_mayout_pixel[0]);
    clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&mem_mayout_pixel[1]);
    clSetKernelArg(kernel, 12, sizeof(cl_mem), (void*)&mem_mayout_pixel[2]);
    clSetKernelArg(kernel, 13, sizeof(channel_info), &mayout_channel[0]);
    clSetKernelArg(kernel, 14, sizeof(channel_info), &mayout_channel[1]);
    clSetKernelArg(kernel, 15, sizeof(channel_info), &mayout_channel[2]);
    clSetKernelArg(kernel, 16, sizeof(cl_int), &clFactor);
    clSetKernelArg(kernel, 17, sizeof(cl_int), &clMask);
    clSetKernelArg(kernel, 18, sizeof(cl_float), &clBlockErrorLimit);
    clSetKernelArg(kernel, 19, sizeof(cl_mem), &mem_output_order_batch);

    size_t globalWorkSize[2] = { blockf_width, blockf_height };
    err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clComputeBlockZeroingOrder() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
    }
    err = clFinish(ocl.commandQueue);
    if (CL_SUCCESS != err)
    {
        LogError("Error: clComputeBlockZeroingOrder() clFinish returned %s.\n", TranslateOpenCLError(err));
    }

    CoeffData *result = (CoeffData *)clEnqueueMapBuffer(ocl.commandQueue, mem_output_order_batch, true, CL_MAP_READ, 0, output_order_batch_size, 0, NULL, NULL, &err);
    err = clFinish(ocl.commandQueue);
    memcpy(output_order_batch, result, output_order_batch_size);

    clEnqueueUnmapMemObject(ocl.commandQueue, mem_output_order_batch, result, 0, NULL, NULL);
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
    size_t xsize, size_t ysize,
    const float* r,  const float* g,  const float* b,
    const float* r2, const float* g2, const float* b2)
{
    cl_int err = CL_SUCCESS;
    ocl_args_d_t &ocl = getOcl();

    cl_int channel_size = xsize * ysize * sizeof(float);

    ocl_channels rgb = ocl.allocMemChannels(channel_size, r, g, b);
    ocl_channels rgb2 = ocl.allocMemChannels(channel_size, r2, g2, b2);
    ocl_channels mask = ocl.allocMemChannels(channel_size);
    ocl_channels mask_dc = ocl.allocMemChannels(channel_size);

    clMaskEx(mask, mask_dc, rgb, rgb2, xsize, ysize);

    cl_float *r0_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mask.r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
    cl_float *r0_g = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mask.g, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
    cl_float *r0_b = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mask.b, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
    cl_float *r1_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mask_dc.r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
    cl_float *r1_g = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mask_dc.g, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
    cl_float *r1_b = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mask_dc.b, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
    err = clFinish(ocl.commandQueue);

    memcpy(mask_r, r0_r, channel_size);
    memcpy(mask_g, r0_g, channel_size);
    memcpy(mask_b, r0_b, channel_size);
    memcpy(maskdc_r, r1_r, channel_size);
    memcpy(maskdc_g, r1_g, channel_size);
    memcpy(maskdc_b, r1_b, channel_size);

    ocl.releaseMemChannels(rgb);
    ocl.releaseMemChannels(rgb2);
    ocl.releaseMemChannels(mask);
    ocl.releaseMemChannels(mask_dc);
}

void clConvolutionEx(
    cl_mem result/*out*/,
    const cl_mem inp, size_t xsize, size_t ysize,
    const cl_mem multipliers, size_t len,
    int xstep, int offset, double border_ratio)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	size_t oxsize = (xsize + xstep - 1) / xstep;

	cl_int clxsize = xsize;
	cl_int clxstep = xstep;
	cl_int cllen = len;
	cl_int cloffset = offset;
	cl_float clborder_ratio = border_ratio;

	cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTION];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&result);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inp);
    clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&clxsize);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&multipliers);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&cllen);
	clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&clxstep);
	clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&cloffset);
	clSetKernelArg(kernel, 7, sizeof(cl_float), (void*)&clborder_ratio);

	size_t globalWorkSize[2] = { oxsize, ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clConvolutionEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clConvolutionEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}
}

void clConvolutionX(
    cl_mem result/*out*/,
    const cl_mem inp, size_t xsize, size_t ysize,
	const cl_mem multipliers, size_t len,
	int xstep, int offset, double border_ratio)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_int clxstep = xstep;
	cl_int cllen = len;
	cl_int cloffset = offset;
	cl_float clborder_ratio = border_ratio;

	cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTIONX];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&result);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inp);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&multipliers);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&cllen);
	clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&xstep);
	clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&cloffset);
	clSetKernelArg(kernel, 6, sizeof(cl_float), (void*)&clborder_ratio);

	size_t globalWorkSize[2] = { xsize, ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clConvolutionEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clConvolutionEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}
}

void clConvolutionY(
    cl_mem result/*out*/,
    const cl_mem inp, size_t xsize, size_t ysize,
	const cl_mem multipliers, size_t len,
	int xstep, int offset, double border_ratio
	)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_int clxstep = xstep;
	cl_int cllen = len;
	cl_int cloffset = offset;
	cl_float clborder_ratio = border_ratio;

	cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTIONY];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&result);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inp);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&multipliers);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&cllen);
	clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&xstep);
	clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&cloffset);
	clSetKernelArg(kernel, 6, sizeof(cl_float), (void*)&clborder_ratio);

	size_t globalWorkSize[2] = { xsize, ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clConvolutionEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clConvolutionEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}
}

void clSquareSampleEx(
    cl_mem result/*out*/,
    const cl_mem image, size_t xsize, size_t ysize,
	size_t xstep, size_t ystep)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_int clxstep = xstep;
	cl_int clystep = ystep;
	cl_kernel kernel = ocl.kernel[KERNEL_SQUARESAMPLE];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&result);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&image);
	clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&clxstep);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&clystep);

	size_t globalWorkSize[2] = { xsize, ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clUpsampleEx clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clUpsampleEx clFinish returned %s.\n", TranslateOpenCLError(err));
	}
}

void clBlurEx(cl_mem image/*out, opt*/, const size_t xsize, const size_t ysize,
    const double sigma, const double border_ratio,
    cl_mem result/*out, opt*/)
{
    clBlurEx2(image, xsize, ysize, sigma, border_ratio, result);

    return;
/*
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
    const int ystep = xstep;
    int dxsize = (xsize + xstep - 1) / xstep;
    int dysize = (ysize + ystep - 1) / ystep;

    cl_int err = 0;
    ocl_args_d_t &ocl = getOcl();
    cl_mem mem_expn = ocl.allocMem(sizeof(cl_float) * expn_size, expn.data());

    if (xstep > 1)
    {
        ocl.allocA(sizeof(cl_float) * dxsize * ysize);
        ocl.allocB(sizeof(cl_float) * dxsize * dysize);

        clConvolutionEx(ocl.srcA, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        clConvolutionEx(ocl.srcB, ocl.srcA, ysize, dxsize, mem_expn, expn_size, ystep, diff, border_ratio);
        clUpsampleEx(result ? result : image, ocl.srcB, xsize, ysize, xstep, ystep);
    }
    else
    {
        ocl.allocA(sizeof(cl_float) * xsize * ysize);
        clConvolutionEx(ocl.srcA, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        clConvolutionEx(result ? result : image, ocl.srcA, ysize, dxsize, mem_expn, expn_size, ystep, diff, border_ratio);
    }

    clReleaseMemObject(mem_expn);
*/
}

void clBlurEx2(cl_mem image/*out, opt*/, size_t xsize, size_t ysize,
	double sigma, double border_ratio,
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

	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	cl_mem mem_expn = ocl.allocMem(sizeof(cl_float) * expn_size, expn.data());

	if (xstep > 1)
	{
		ocl.allocA(sizeof(cl_float) * xsize * ysize);
		clConvolutionX(ocl.srcA, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
		clConvolutionY(result ? result : image, ocl.srcA, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
        clSquareSampleEx(result ? result : image, result ? result : image, xsize, ysize, xstep, xstep);
	}
	else
	{
		ocl.allocA(sizeof(cl_float) * xsize * ysize);
		clConvolutionX(ocl.srcA, image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
		clConvolutionY(result ? result : image, ocl.srcA, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
	}

	clReleaseMemObject(mem_expn);
}

void clOpsinDynamicsImageEx(ocl_channels &rgb, const size_t xsize, const size_t ysize)
{
	static const double kSigma = 1.1;

	cl_int channel_size = xsize * ysize * sizeof(float);

	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels rgb_blurred = ocl.allocMemChannels(channel_size);

	clBlurEx(rgb.r, xsize, ysize, kSigma, 0.0, rgb_blurred.r);
	clBlurEx(rgb.g, xsize, ysize, kSigma, 0.0, rgb_blurred.g);
	clBlurEx(rgb.b, xsize, ysize, kSigma, 0.0, rgb_blurred.b);

	cl_kernel kernel = ocl.kernel[KERNEL_OPSINDYNAMICSIMAGE];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&rgb.r);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&rgb.g);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&rgb.b);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&rgb_blurred.r);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&rgb_blurred.g);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&rgb_blurred.b);

	size_t globalWorkSize[1] = { xsize * ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clOpsinDynamicsImageEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clOpsinDynamicsImageEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}

	ocl.releaseMemChannels(rgb_blurred);
}


void clMaskHighIntensityChangeEx(
    ocl_channels &xyb0/*in,out*/,
    ocl_channels &xyb1/*in,out*/,
    const size_t xsize, const size_t ysize)
{
	cl_int channel_size = xsize * ysize * sizeof(float);

	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	ocl_channels c0 = ocl.allocMemChannels(channel_size);
	ocl_channels c1 = ocl.allocMemChannels(channel_size);

	clEnqueueCopyBuffer(ocl.commandQueue, xyb0.r, c0.r, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb0.g, c0.g, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb0.b, c0.b, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb1.r, c1.r, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb1.g, c1.g, 0, 0, channel_size, 0, NULL, NULL);
	clEnqueueCopyBuffer(ocl.commandQueue, xyb1.b, c1.b, 0, 0, channel_size, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	cl_kernel kernel = ocl.kernel[KERNEL_MASKHIGHINTENSITYCHANGE];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&xyb0.r);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&xyb0.g);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&xyb0.b);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&xyb1.r);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&xyb1.g);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&xyb1.b);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&c0.r);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&c0.g);
	clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&c0.b);
	clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&c1.r);
	clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&c1.g);
	clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&c1.b);

	size_t globalWorkSize[2] = { xsize, ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clMaskHighIntensityChangeEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clMaskHighIntensityChangeEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}

	ocl.releaseMemChannels(c0);
	ocl.releaseMemChannels(c1);
}

void clEdgeDetectorMapEx(
    cl_mem result/*out*/,
    const ocl_channels &rgb, const ocl_channels &rgb2,
    const size_t xsize, const size_t ysize, const size_t step)
{
	cl_int channel_size = xsize * ysize * sizeof(float);

	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	ocl_channels rgb_blured = ocl.allocMemChannels(channel_size);
	ocl_channels rgb2_blured = ocl.allocMemChannels(channel_size);

	static const double kSigma[3] = { 1.5, 0.586, 0.4 };

	for (int i = 0; i < 3; i++)
	{
		clBlurEx(rgb.ch[i], xsize, ysize, kSigma[i], 0.0, rgb_blured.ch[i]);
		clBlurEx(rgb2.ch[i], xsize, ysize, kSigma[i], 0.0, rgb2_blured.ch[i]);
	}

	cl_int clxsize = xsize;
	cl_int clysize = ysize;
	cl_int clstep = step;

	cl_kernel kernel = ocl.kernel[KERNEL_EDGEDETECTOR];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &result);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &rgb_blured.r);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &rgb_blured.g);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &rgb_blured.b);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &rgb2_blured.r);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &rgb2_blured.g);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), &rgb2_blured.b);
	clSetKernelArg(kernel, 7, sizeof(cl_int), &clxsize);
	clSetKernelArg(kernel, 8, sizeof(cl_int), &clysize);
	clSetKernelArg(kernel, 9, sizeof(cl_int), &clstep);

	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;

	size_t globalWorkSize[2] = { res_xsize, res_ysize};
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEdgeDetectorMapEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEdgeDetectorMapEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}

	ocl.releaseMemChannels(rgb_blured);
	ocl.releaseMemChannels(rgb2_blured);
}

void clBlockDiffMapEx(
    cl_mem block_diff_dc/*out*/,
    cl_mem block_diff_ac/*out*/,
    const ocl_channels &rgb, const ocl_channels &rgb2,
	const size_t xsize, const size_t ysize, const size_t step)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_int clxsize = xsize;
	cl_int clysize = ysize;
	cl_int clstep = step;

	cl_kernel kernel = ocl.kernel[KERNEL_BLOCKDIFFMAP];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &block_diff_dc);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &block_diff_ac);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &rgb.r);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &rgb.g);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &rgb.b);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &rgb2.r);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), &rgb2.g);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), &rgb2.b);
	clSetKernelArg(kernel, 8, sizeof(cl_int), &clxsize);
	clSetKernelArg(kernel, 9, sizeof(cl_int), &clysize);
	clSetKernelArg(kernel, 10, sizeof(cl_int), &clstep);

	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;

	size_t globalWorkSize[2] = { res_xsize, res_ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBlockDiffMapEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBlockDiffMapEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}
}

void clEdgeDetectorLowFreqEx(
    cl_mem block_diff_ac/*in,out*/,
    const ocl_channels &rgb, const ocl_channels &rgb2,
	const size_t xsize, const size_t ysize, const size_t step)
{
	cl_int channel_size = xsize * ysize * sizeof(float);

	static const double kSigma = 14;

	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels rgb_blured = ocl.allocMemChannels(channel_size);
	ocl_channels rgb2_blured = ocl.allocMemChannels(channel_size);

	for (int i = 0; i < 3; i++)
	{
		clBlurEx(rgb.ch[i], xsize, ysize, kSigma, 0.0, rgb_blured.ch[i]);
		clBlurEx(rgb2.ch[i], xsize, ysize, kSigma, 0.0, rgb2_blured.ch[i]);
	}

	cl_int clxsize = xsize;
	cl_int clysize = ysize;
	cl_int clstep = step;

	cl_kernel kernel = ocl.kernel[KERNEL_EDGEDETECTORLOWFREQ];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &block_diff_ac);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &rgb_blured.r);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &rgb_blured.g);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &rgb_blured.b);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &rgb2_blured.r);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &rgb2_blured.g);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), &rgb2_blured.b);
	clSetKernelArg(kernel, 7, sizeof(cl_int), &clxsize);
	clSetKernelArg(kernel, 8, sizeof(cl_int), &clysize);
	clSetKernelArg(kernel, 9, sizeof(cl_int), &clstep);

	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;

	size_t globalWorkSize[2] = { res_xsize, res_ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEdgeDetectorLowFreqEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEdgeDetectorLowFreqEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}

	ocl.releaseMemChannels(rgb_blured);
	ocl.releaseMemChannels(rgb2_blured);
}

void clDiffPrecomputeEx(
    ocl_channels &mask/*out*/,
    const ocl_channels &xyb0, const ocl_channels &xyb1,
    const size_t xsize, const size_t ysize)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_kernel kernel = ocl.kernel[KERNEL_DIFFPRECOMPUTE];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mask.x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mask.y);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mask.b);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&xyb0.x);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&xyb0.y);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&xyb0.b);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&xyb1.x);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&xyb1.y);
	clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&xyb1.b);

	size_t globalWorkSize[2] = { xsize, ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clDiffPrecomputeEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clDiffPrecomputeEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}
}

void clScaleImageEx(cl_mem img/*in, out*/, size_t size, double w)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_double clscale = w;

	cl_kernel kernel = ocl.kernel[KERNEL_SCALEIMAGE];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&img);
	clSetKernelArg(kernel, 1, sizeof(cl_double), (void*)&clscale);

	size_t globalWorkSize[1] = { size };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clScaleImageEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clScaleImageEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}
}

void clAverage5x5Ex(cl_mem img/*in,out*/, const size_t xsize, const size_t ysize)
{
    if (xsize < 4 || ysize < 4) {
	    // TODO: Make this work for small dimensions as well.
	    return;
    }

    cl_int err = CL_SUCCESS;
    ocl_args_d_t &ocl = getOcl();

    size_t len = xsize * ysize * sizeof(float);
    ocl.allocA(len);
    cl_mem img_org = ocl.srcA;

    err = clEnqueueCopyBuffer(ocl.commandQueue, img, img_org, 0, 0, len, 0, NULL, NULL);

    cl_kernel kernel = ocl.kernel[KERNEL_AVERAGE5X5];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&img);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&img_org);

    size_t globalWorkSize[2] = { xsize, ysize };
    err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    if (CL_SUCCESS != err)
    {
    LogError("Error: clAverage5x5Ex() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
    }
    err = clFinish(ocl.commandQueue);
    if (CL_SUCCESS != err)
    {
    LogError("Error: clAverage5x5Ex() clFinish returned %s.\n", TranslateOpenCLError(err));
    }
}

void clMinSquareValEx(
    cl_mem img/*in,out*/,
    const size_t xsize, const size_t ysize,
    const size_t square_size, const size_t offset)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_int cloffset = offset;
	cl_int clsquare_size = square_size;
	ocl.allocA(sizeof(cl_float) * xsize * ysize);

	cl_kernel kernel = ocl.kernel[KERNEL_MINSQUAREVAL];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&ocl.srcA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&img);
	clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&clsquare_size);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&cloffset);

	size_t globalWorkSize[2] = { xsize, ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clMinSquareValEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}

	err = clEnqueueCopyBuffer(ocl.commandQueue, ocl.srcA, img, 0, 0, sizeof(cl_float) * xsize * ysize, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clMinSquareValEx() clEnqueueCopyBuffer returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clMinSquareValEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}
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
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_int clxsize = xsize;
	cl_int clysize = ysize;

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

	size_t channel_size = 512 * 3 * sizeof(double);
	ocl_channels xyb = ocl.allocMemChannels(channel_size, lut_x, lut_y, lut_b);
    ocl_channels xyb_dc = ocl.allocMemChannels(channel_size, lut_dcx, lut_dcy, lut_dcb);

	cl_kernel kernel = ocl.kernel[KERNEL_DOMASK];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mask.r);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mask.g);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mask.b);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&mask_dc.r);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&mask_dc.g);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&mask_dc.b);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&xyb.x);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&xyb.y);
	clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&xyb.b);
	clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&xyb_dc.x);
	clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&xyb_dc.y);
	clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&xyb_dc.b);

	size_t globalWorkSize[2] = { xsize, ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clDoMask() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clDoMask() clFinish returned %s.\n", TranslateOpenCLError(err));
	}

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
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	const size_t work_xsize = ((xsize - 8 + step) + step - 1) / step;
	const size_t work_ysize = ((ysize - 8 + step) + step - 1) / step;

	cl_int clres_size = res_xsize;
	cl_int clxsize = xsize;
	cl_int clysize = ysize;
	cl_int clstep = step;

	cl_kernel kernel = ocl.kernel[KERNEL_COMBINECHANNELS];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&result);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mask.r);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mask.g);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&mask.b);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&mask_dc.r);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&mask_dc.g);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&mask_dc.b);
    clSetKernelArg(kernel, 7, sizeof(cl_int), (void*)&clxsize);
    clSetKernelArg(kernel, 8, sizeof(cl_int), (void*)&clysize);
	clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&block_diff_dc);
	clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&block_diff_ac);
	clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&edge_detector_map);
	clSetKernelArg(kernel, 12, sizeof(cl_int), (void*)&clres_size);
	clSetKernelArg(kernel, 13, sizeof(cl_int), (void*)&clstep);

	size_t globalWorkSize[2] = { work_xsize, work_ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCombineChannelsEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCombineChannelsEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}
}

void clUpsampleSquareRootEx(cl_mem diffmap, const size_t xsize, const size_t ysize, const int step)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_int clxsize = xsize;
	cl_int clysize = ysize;
	cl_int clstep = step;

    cl_mem diffmap_out = ocl.allocMem(xsize * ysize * sizeof(float));

	cl_kernel kernel = ocl.kernel[KERNEL_UPSAMPLESQUAREROOT];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&diffmap_out);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&diffmap);
	clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&xsize);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&ysize);
	clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&step);

	const size_t res_xsize = (xsize + step - 1) / step;
	const size_t res_ysize = (ysize + step - 1) / step;

	size_t globalWorkSize[2] = { res_xsize, res_ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clUpsampleSquareRootEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	err = clEnqueueCopyBuffer(ocl.commandQueue, diffmap_out, diffmap, 0, 0, xsize * ysize * sizeof(float), 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clUpsampleSquareRootEx() clEnqueueCopyBuffer returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clUpsampleSquareRootEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}

    clReleaseMemObject(diffmap_out);
}

void clRemoveBorderEx(cl_mem out, const cl_mem in, const size_t xsize, const size_t ysize, const int step)
{
	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();

	cl_int cls = 8 - step;
	cl_int cls2 = (8 - step) / 2;
    cl_int clxsize = xsize;
	cl_kernel kernel = ocl.kernel[KERNEL_REMOVEBORDER];
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &out);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &in);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &clxsize);
	clSetKernelArg(kernel, 3, sizeof(cl_int), &cls);
	clSetKernelArg(kernel, 4, sizeof(cl_int), &cls2);

	size_t globalWorkSize[2] = { xsize - cls, ysize - cls};
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCalculateDiffmapGetBlurredEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCalculateDiffmapGetBlurredEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}
}

void clAddBorderEx(cl_mem out, size_t xsize, size_t ysize, int step, cl_mem in)
{
	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();

    cl_int cls = 8 - step;
    cl_int cls2 = (8 - step) / 2;
	cl_kernel kernel = ocl.kernel[KERNEL_ADDBORDER];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out);
	clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&cls);
	clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&cls2);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&in);

	size_t globalWorkSize[2] = { xsize, ysize};
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDiffmapFromBlurredEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDiffmapFromBlurredEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}
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

void cuScaleImage(float *img, size_t length, double scale)
{
	ocu_args_d_t &ocu = getOcu();
	CUdeviceptr m = ocu.allocMem(length * sizeof(float), img);

	void *args[2] = { &m, &scale};

	CUresult r = cuLaunchKernel(ocu.kernel[KERNEL_SCALEIMAGE],
                   1, 1, 1,
                   length, 1, 1,
                   0,
                   ocu.stream, args, NULL);

    r = cuStreamSynchronize(ocu.stream);

    cuMemcpyDtoH(img, m, length * sizeof(float));

	cuMemFree(m);
	return;
}