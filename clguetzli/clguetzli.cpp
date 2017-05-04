#include <math.h>
#include <algorithm>
#include <vector>
#include "clguetzli.h"
#include "ocl.h"

extern bool g_useOpenCL = false;

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
	ReadSourceFromFile("clguetzli.cl", &source, &src_size);

	ocl.program = clCreateProgramWithSource(ocl.context, 1, (const char**)&source, &src_size, &err);

	delete[] source;

	err = clBuildProgram(ocl.program, 1, &ocl.device, "", NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));
	}
	ocl.kernel[KERNEL_MINSQUAREVAL] = clCreateKernel(ocl.program, "MinSquareVal", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel(MinSquareVal) for source program returned %s.\n", TranslateOpenCLError(err));
	}
	ocl.kernel[KERNEL_CONVOLUTION] = clCreateKernel(ocl.program, "Convolution", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateKernel(Convolution) for source program returned %s.\n", TranslateOpenCLError(err));
	}
	ocl.kernel[KERNEL_CONVOLUTIONX] = clCreateKernel(ocl.program, "ConvolutionX", &err);
	ocl.kernel[KERNEL_CONVOLUTIONY] = clCreateKernel(ocl.program, "ConvolutionY", &err);
	ocl.kernel[KERNEL_DOWNSAMPLE] = clCreateKernel(ocl.program, "DownSample", &err);

	return ocl;
}

void clMinSquareVal(size_t square_size, size_t offset,
	size_t xsize, size_t ysize,
	float *values)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	ocl.allocA(sizeof(cl_float) * xsize * ysize);
	ocl.allocC(sizeof(cl_float) * xsize * ysize);

	memcpy(ocl.inputA, values, sizeof(cl_float) * xsize * ysize);

	cl_int cloffset = offset;
	cl_int clsquare_size = square_size;

	cl_kernel kernel = ocl.kernel[KERNEL_MINSQUAREVAL];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&ocl.srcA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&ocl.dstMem);
	clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&clsquare_size);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&cloffset);

	size_t globalWorkSize[2] = { xsize, ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));
	}

	cl_float *resultPtr = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, ocl.dstMem, true, CL_MAP_READ, 0, sizeof(cl_float) * xsize * ysize, 0, NULL, NULL, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueMapBuffer returned %s\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clFinish returned %s\n", TranslateOpenCLError(err));
	}

	memcpy(values, resultPtr, sizeof(cl_float) * xsize * ysize);
}

void clConvolution(size_t xsize, size_t ysize,
	size_t xstep,
	size_t len, size_t offset,
	const float* multipliers,
	const float* inp,
	float border_ratio,
	float* result)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	size_t oxsize = xsize / xstep;

	ocl.allocA(sizeof(cl_float) * len);
	ocl.allocB(sizeof(cl_float) * xsize * ysize);
	ocl.allocC(sizeof(cl_float) * oxsize * ysize);
	
	memcpy(ocl.inputA, multipliers, sizeof(cl_float) * len);
	memcpy(ocl.inputB, inp, sizeof(cl_float) * xsize * ysize);

	cl_int clxsize = xsize;
	cl_int clxstep = xstep;
	cl_int cllen = len;
	cl_int cloffset = offset;
	cl_float clborder_ratio = border_ratio;

	cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTION];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&ocl.srcA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&ocl.srcB);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&ocl.dstMem);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&clxsize);
	clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&clxstep);
	clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&cllen);
	clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&cloffset);
	clSetKernelArg(kernel, 7, sizeof(cl_float), (void*)&clborder_ratio);

	size_t globalWorkSize[2] = { xsize / xstep, ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));
	}

	cl_float *resultPtr = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, ocl.dstMem, true, CL_MAP_READ, 0, sizeof(cl_float) * oxsize * ysize, 0, NULL, NULL, &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clEnqueueMapBuffer returned %s\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clFinish returned %s\n", TranslateOpenCLError(err));
	}

	memcpy(result, resultPtr, sizeof(cl_float) * oxsize * ysize);
}

void clBlur(size_t xsize, size_t ysize, float* channel, double sigma, double border_ratio)
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

	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	ocl.allocA(sizeof(cl_float) * expn_size);
	ocl.allocB(sizeof(cl_float) * xsize * ysize);
	ocl.allocC(sizeof(cl_float) * xsize * ysize);

	memcpy(ocl.inputA, expn.data(), sizeof(cl_float) * expn_size);
	memcpy(ocl.inputB, channel, sizeof(cl_float) * xsize * ysize);

	cl_int clxsize = xsize;
	cl_int clxstep = xstep;
	cl_int cllen = expn_size;
	cl_int cloffset = diff;
	cl_float clborder_ratio = border_ratio;

	cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTION];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&ocl.srcA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&ocl.srcB);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&ocl.dstMem);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&clxsize);
	clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&clxstep);
	clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&cllen);
	clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&cloffset);
	clSetKernelArg(kernel, 7, sizeof(cl_float), (void*)&clborder_ratio);

	size_t globalWorkSize[2] = { xsize / xstep, ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	globalWorkSize[0] = ysize / xstep;
	globalWorkSize[1] = xsize / xstep;
	clxsize = ysize;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&ocl.srcA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&ocl.dstMem);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&ocl.srcB);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&clxsize);
	clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&clxstep);
	clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&cllen);
	clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&cloffset);
	clSetKernelArg(kernel, 7, sizeof(cl_float), (void*)&clborder_ratio);

	err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	cl_int clstep = xstep;
	if (clstep <= 1)
	{
		cl_float *resultPtr = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, ocl.srcB, true, CL_MAP_READ, 0, sizeof(cl_float) * xsize * ysize, 0, NULL, NULL, &err);
		err = clFinish(ocl.commandQueue);
		memcpy(channel, resultPtr, sizeof(cl_float) * xsize * ysize);
	}
	else
	{
		kernel = ocl.kernel[KERNEL_DOWNSAMPLE];
		clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&ocl.srcB);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&ocl.dstMem);
		clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&clstep);

		globalWorkSize[0] = ysize;
		globalWorkSize[1] = xsize;
		err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
		err = clFinish(ocl.commandQueue);

		cl_float *resultPtr = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, ocl.dstMem, true, CL_MAP_READ, 0, sizeof(cl_float) * xsize * ysize, 0, NULL, NULL, &err);
		err = clFinish(ocl.commandQueue);
		memcpy(channel, resultPtr, sizeof(cl_float) * xsize * ysize);
	}
}

void clConvolutionEx(cl_mem image, size_t xsize, size_t ysize, cl_mem expn, size_t expn_size, 
	int step, int offset, double border_ratio, cl_mem result)
{
	// Convolution
}

void clUpsampleEx(cl_mem image, size_t xsize, size_t ysize, size_t xstep, size_t ystep, cl_mem result)
{
/*
	for (size_t y = 0; y < ysize; y++) {
		for (size_t x = 0; x < xsize; x++) {
			// TODO: Use correct rounding.
			channel[y * xsize + x] =
				downsampled_output[(y / ystep) * dxsize + (x / xstep)];
		}
	}
*/
}

void clBlurEx(cl_mem image, size_t xsize, size_t ysize, double sigma, double border_ratio)
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
	const int ystep = xstep;
	int dxsize = (xsize + xstep - 1) / xstep;
	int dysize = (ysize + ystep - 1) / ystep;

	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	cl_mem mem_expn = ocl.allocMem(sizeof(cl_float) * expn_size);

	clEnqueueWriteBuffer(ocl.commandQueue, mem_expn, CL_FALSE, 0, sizeof(cl_float) * expn_size, expn.data(), 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	if (xstep > 1)
	{
		ocl.allocA(sizeof(cl_float) * dxsize * ysize);
		ocl.allocB(sizeof(cl_float) * dxsize * dysize);

		clConvolutionEx(image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio, ocl.srcA);
		clConvolutionEx(ocl.srcA, ysize, dxsize, mem_expn, expn_size, ystep, diff, border_ratio, ocl.srcB);
		clUpsampleEx(ocl.srcB, dxsize, dysize, xstep, ystep, image);
	}
	else
	{
		ocl.allocA(sizeof(cl_float) * xsize * ysize);
		clConvolutionEx(image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio, ocl.srcA);
		clConvolutionEx(ocl.srcA, ysize, dxsize, mem_expn, expn_size, ystep, diff, border_ratio, image);
	}

	clReleaseMemObject(mem_expn);
}

void clOpsinDynamicsImageEx(cl_mem r, cl_mem g, cl_mem b, size_t size)
{
/*
	for (size_t i = 0; i < rgb[0].size(); ++i) {
		double sensitivity[3];
		{
			// Calculate sensitivity[3] based on the smoothed image gamma derivative.
			double pre_rgb[3] = { blurred[0][i], blurred[1][i], blurred[2][i] };
			double pre_mixed[3];
			OpsinAbsorbance(pre_rgb, pre_mixed);
			sensitivity[0] = Gamma(pre_mixed[0]) / pre_mixed[0];
			sensitivity[1] = Gamma(pre_mixed[1]) / pre_mixed[1];
			sensitivity[2] = Gamma(pre_mixed[2]) / pre_mixed[2];
		}
		double cur_rgb[3] = { rgb[0][i],  rgb[1][i],  rgb[2][i] };
		double cur_mixed[3];
		OpsinAbsorbance(cur_rgb, cur_mixed);
		cur_mixed[0] *= sensitivity[0];
		cur_mixed[1] *= sensitivity[1];
		cur_mixed[2] *= sensitivity[2];
		double x, y, z;
		RgbToXyb(cur_mixed[0], cur_mixed[1], cur_mixed[2], &x, &y, &z);
		rgb[0][i] = static_cast<float>(x);
		rgb[1][i] = static_cast<float>(y);
		rgb[2][i] = static_cast<float>(z);
*/
}

void clOpsinDynamicsImage(size_t xsize, size_t ysize, float* r, float* g, float* b)
{
	static const double kSigma = 1.1;

	cl_int channel_size = xsize * ysize * sizeof(float);
	
	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	cl_mem mem_r = ocl.allocMem(channel_size);
	cl_mem mem_g = ocl.allocMem(channel_size);
	cl_mem mem_b = ocl.allocMem(channel_size);

	clEnqueueWriteBuffer(ocl.commandQueue, mem_r, CL_FALSE, 0, channel_size, r, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, mem_g, CL_FALSE, 0, channel_size, g, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, mem_b, CL_FALSE, 0, channel_size, b, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clBlurEx(mem_r, xsize, ysize, kSigma, 0.0);
	clBlurEx(mem_g, xsize, ysize, kSigma, 0.0);
	clBlurEx(mem_b, xsize, ysize, kSigma, 0.0);

	clOpsinDynamicsImageEx(mem_r, mem_g, mem_b, xsize * ysize);

	cl_float *result_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mem_r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *result_g = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mem_g, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *result_b = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mem_b, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	err = clFinish(ocl.commandQueue);

	memcpy(r, result_r, channel_size);
	memcpy(g, result_g, channel_size);
	memcpy(b, result_b, channel_size);

	clReleaseMemObject(mem_r);
	clReleaseMemObject(mem_g);
	clReleaseMemObject(mem_b);
}

void clMaskHighIntensityChangeEx(ocl_channels rgb, ocl_channels rgb2, size_t xsize, size_t ysize)
{
	// MaskHighIntensityChange
}

void clEdgeDetectorMapEx(ocl_channels rgb, ocl_channels rgb2, size_t xsize, size_t ysize, cl_mem result)
{
	static const double kSigma[3] = { 1.5, 0.586, 0.4 };
	clBlurEx(rgb.r,  xsize, ysize, kSigma[0], 0.0);
	clBlurEx(rgb2.r, xsize, ysize, kSigma[0], 0.0);
	clBlurEx(rgb.g,  xsize, ysize, kSigma[1], 0.0);
	clBlurEx(rgb2.g, xsize, ysize, kSigma[1], 0.0);
	clBlurEx(rgb.b,  xsize, ysize, kSigma[2], 0.0);
	clBlurEx(rgb2.b, xsize, ysize, kSigma[2], 0.0);

	// EdgeDetectorLowFreq
}

void clBlockDiffMapEx(ocl_channels rgb, ocl_channels rgb2,
	size_t xsize, size_t ysize,
	cl_mem block_diff_dc, cl_mem block_diff_ac)
{

}

void clEdgeDetectorLowFreqEx(ocl_channels rgb, ocl_channels rgb2,
	size_t xsize, size_t ysize,
	cl_mem block_diff_ac)
{

}

void clMaskEx(ocl_channels rgb, ocl_channels rgb2,
	size_t xsize, size_t ysize,
	ocl_channels mask, ocl_channels mask_dc)
{

}

void clCombineChannelsEx(ocl_channels mask, ocl_channels mask_dc, cl_mem block_diff_dc, cl_mem block_diff_ac, cl_mem edge_detector_map, cl_mem result)
{

}

void clCalculateDiffmap(cl_mem result, size_t xsize, size_t ysize, int step)
{

}

void clDiffmapOpsinDynamicsImage(float* r, float* g, float* b, 
								 float* r2, float* g2, float* b2, 
								 size_t xsize, size_t ysize,
								 float* result)
{

	cl_int channel_size = xsize * ysize * sizeof(float);

	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels xyb = ocl.allocMemChannels(channel_size);
	ocl_channels xyb2 = ocl.allocMemChannels(channel_size);

	clEnqueueWriteBuffer(ocl.commandQueue, xyb.r, CL_FALSE, 0, channel_size, r, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb.g, CL_FALSE, 0, channel_size, g, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb.b, CL_FALSE, 0, channel_size, b, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb2.r, CL_FALSE, 0, channel_size, r2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb2.g, CL_FALSE, 0, channel_size, g2, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, xyb2.b, CL_FALSE, 0, channel_size, b2, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clMaskHighIntensityChangeEx(xyb, xyb2, xsize, ysize);

	cl_mem edge_detector_map = ocl.allocMem(3 * xsize * ysize);
	cl_mem block_diff_dc = ocl.allocMem(3 * xsize * ysize);
	cl_mem block_diff_ac = ocl.allocMem(3 * xsize * ysize);

	ocl_channels mask;
	ocl_channels mask_dc;
	
	cl_mem mem_result;

	clEdgeDetectorMapEx(xyb, xyb2, xsize, ysize, edge_detector_map);
	clBlockDiffMapEx(xyb, xyb2, xsize, ysize, block_diff_dc, block_diff_ac);
	clEdgeDetectorLowFreqEx(xyb, xyb2, xsize, ysize, block_diff_ac);
	
	clMaskEx(xyb, xyb2, xsize, ysize, mask, mask_dc);
	clCombineChannelsEx(mask, mask_dc, block_diff_dc, block_diff_ac, edge_detector_map, mem_result);
}
