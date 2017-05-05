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

        if (err == CL_BUILD_PROGRAM_FAILURE)
        {
            size_t log_size = 0;
            clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

            std::vector<char> build_log(log_size);
            clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], NULL);

            LogError("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
        }
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
	ocl.kernel[KERNEL_OPSINDYNAMICSIMAGE] = clCreateKernel(ocl.program, "OpsinDynamicsImage", &err);
	ocl.kernel[KERNEL_DOMASK] = clCreateKernel(ocl.program, "DoMask", &err);
	ocl.kernel[KERNEL_SCALEIMAGE] = clCreateKernel(ocl.program, "ScaleImage", &err);
	ocl.kernel[KERNEL_COMBINECHANNELS] = clCreateKernel(ocl.program, "CombineChannels", &err);

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
		clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&clstep);

		globalWorkSize[0] = ysize;
		globalWorkSize[1] = xsize;
		err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
		err = clFinish(ocl.commandQueue);

		cl_float *resultPtr = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, ocl.dstMem, true, CL_MAP_READ, 0, sizeof(cl_float) * xsize * ysize, 0, NULL, NULL, &err);
		err = clFinish(ocl.commandQueue);
		memcpy(channel, resultPtr, sizeof(cl_float) * xsize * ysize);
	}
}

void clConvolutionEx(cl_mem inp, size_t xsize, size_t ysize,
				     cl_mem multipliers, size_t len,
                     int xstep, int offset, double border_ratio, 
                     cl_mem result/*out*/)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	size_t oxsize = xsize / xstep;

	cl_int clxsize = xsize;
	cl_int clxstep = xstep;
	cl_int cllen = len;
	cl_int cloffset = offset;
	cl_float clborder_ratio = border_ratio;

	cl_kernel kernel = ocl.kernel[KERNEL_CONVOLUTION];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&multipliers);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&inp);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&result);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&clxsize);
	clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&clxstep);
	clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&cllen);
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

void clUpsampleEx(cl_mem image, size_t xsize, size_t ysize, 
                  size_t xstep, size_t ystep, 
                  cl_mem result/*out*/)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_int clxstep = xstep;
	cl_int clystep = ystep;
	cl_kernel kernel = ocl.kernel[KERNEL_DOWNSAMPLE];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&ocl.srcB);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&ocl.dstMem);
	clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&clxstep);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&clystep);

	size_t globalWorkSize[2] = { ysize, xsize };
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

void clBlurEx(cl_mem image/*out, opt*/, size_t xsize, size_t ysize, 
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
		clUpsampleEx(ocl.srcB, dxsize, dysize, xstep, ystep, result ? result : image);
	}
	else
	{
		ocl.allocA(sizeof(cl_float) * xsize * ysize);
		clConvolutionEx(image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio, ocl.srcA);
		clConvolutionEx(ocl.srcA, ysize, dxsize, mem_expn, expn_size, ystep, diff, border_ratio, result ? result : image);
	}

	clReleaseMemObject(mem_expn);
}

void clOpsinDynamicsImageEx(ocl_channels rgb/*in,out*/, ocl_channels rgb_blurred, size_t size)
{
	ocl_args_d_t &ocl = getOcl();
	cl_int clSize = size;
	cl_kernel kernel = ocl.kernel[KERNEL_OPSINDYNAMICSIMAGE];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&rgb.r);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&rgb.g);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&rgb.b);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&rgb_blurred.r);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&rgb_blurred.g);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&rgb_blurred.b);
	clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&clSize);

	size_t globalWorkSize[1] = { clSize };
	cl_int err = clEnqueueNDRangeKernel(ocl.commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clOpsinDynamicsImageEx() clEnqueueNDRangeKernel returned %s.\n", TranslateOpenCLError(err));
	}
	err = clFinish(ocl.commandQueue);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clOpsinDynamicsImageEx() clFinish returned %s.\n", TranslateOpenCLError(err));
	}
}

void clOpsinDynamicsImage(size_t xsize, size_t ysize, float* r, float* g, float* b)
{
	static const double kSigma = 1.1;

	cl_int channel_size = xsize * ysize * sizeof(float);
	
	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
    ocl_channels rgb = ocl.allocMemChannels(channel_size);
	ocl_channels rgb_blurred = ocl.allocMemChannels(channel_size);

	clEnqueueWriteBuffer(ocl.commandQueue, rgb.r, CL_FALSE, 0, channel_size, r, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, rgb.g, CL_FALSE, 0, channel_size, g, 0, NULL, NULL);
	clEnqueueWriteBuffer(ocl.commandQueue, rgb.b, CL_FALSE, 0, channel_size, b, 0, NULL, NULL);
	err = clFinish(ocl.commandQueue);

	clBlurEx(rgb.r, xsize, ysize, kSigma, 0.0, rgb_blurred.r);
	clBlurEx(rgb.g, xsize, ysize, kSigma, 0.0, rgb_blurred.g);
	clBlurEx(rgb.b, xsize, ysize, kSigma, 0.0, rgb_blurred.b);

	clOpsinDynamicsImageEx(rgb, rgb_blurred, xsize * ysize);

	cl_float *result_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, rgb.r, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *result_g = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, rgb.g, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	cl_float *result_b = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, rgb.b, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);

	err = clFinish(ocl.commandQueue);

	memcpy(r, result_r, channel_size);
	memcpy(g, result_g, channel_size);
	memcpy(b, result_b, channel_size);

    ocl.releaseMemChannels(rgb);
	ocl.releaseMemChannels(rgb_blurred);
}

// ian todo
void clMaskHighIntensityChangeEx(ocl_channels rgb/*in,out*/, 
                                 ocl_channels rgb2/*in,out*/, 
                                 size_t xsize, size_t ysize)
{
	// MaskHighIntensityChange
}

// strong todo
void clEdgeDetectorMapEx(ocl_channels rgb, ocl_channels rgb2, size_t xsize, size_t ysize, size_t step, cl_mem result/*out*/)
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
	// EdgeDetectorLowFreq

}

// strong todo
void clBlockDiffMapEx(ocl_channels rgb, ocl_channels rgb2,
	size_t xsize, size_t ysize,
	cl_mem block_diff_dc/*out*/, cl_mem block_diff_ac/*out*/)
{

}

// strong todo
void clEdgeDetectorLowFreqEx(ocl_channels rgb, ocl_channels rgb2,
	size_t xsize, size_t ysize,
	cl_mem block_diff_ac/*out*/)
{
	cl_int channel_size = xsize * ysize * sizeof(float);

	static const double kSigma = 14;
	static const double kMul = 10;

	cl_int err = 0;
	ocl_args_d_t &ocl = getOcl();
	ocl_channels rgb_blured = ocl.allocMemChannels(channel_size);
	ocl_channels rgb2_blured = ocl.allocMemChannels(channel_size);

	static const double kSigma[3] = { 1.5, 0.586, 0.4 };

	for (int i = 0; i < 3; i++)
	{
		clBlurEx(rgb.ch[i], xsize, ysize, kSigma, 0.0, rgb_blured.ch[i]);
		clBlurEx(rgb2.ch[i], xsize, ysize, kSigma, 0.0, rgb2_blured.ch[i]);
	}
}

// ian todo
void clDiffPrecomputeEx(ocl_channels rgb, ocl_channels rgb2, size_t xsize, size_t ysize, ocl_channels mask/*out*/)
{

}

void clScaleImageEx(cl_mem img, size_t size, float w, cl_mem result/*out*/)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_int clsize = size;
	cl_float clscale = w;


	err = clEnqueueCopyBuffer(ocl.commandQueue, img, result, 0, 0, clsize, 0, NULL, NULL);	
	if (CL_SUCCESS != err)
	{
		LogError("Error: clScaleImageEx() clEnqueueCopyBuffer returned %s.\n", TranslateOpenCLError(err));
	}

	cl_kernel kernel = ocl.kernel[KERNEL_SCALEIMAGE];
	clSetKernelArg(kernel, 0, sizeof(cl_int), (void*)&clscale);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&result);

	size_t globalWorkSize[1] = { clsize };
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

// ian todo
void clAverage5x5Ex(cl_mem img/*in,out*/, size_t xsize, size_t ysize)
{
    static const float w = 0.679144890667f;
    static const float scale = 1.0f / (5.0f + 4 * w);

    cl_mem tmp0;
    cl_mem tmp1;
    clScaleImageEx(img, xsize * ysize, w, tmp0);
    clScaleImageEx(img, xsize * ysize, 1, tmp1);
    // average5x5 calc

    clScaleImageEx(img, xsize * ysize, scale, img);
}

void clMinSquareValEx(cl_mem img/*in,out*/, size_t xsize, size_t ysize, size_t square_size, size_t offset)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_int cloffset = offset;
	cl_int clsquare_size = square_size;
	ocl.allocA(sizeof(cl_float) * xsize * ysize);

	cl_kernel kernel = ocl.kernel[KERNEL_MINSQUAREVAL];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&img);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&ocl.srcA);
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

static const double kInternalGoodQualityThreshold = 14.921561160295326;
static const double kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

void clDoMask(ocl_channels mask/*in, out*/, ocl_channels mask_dc/*in, out*/, size_t xsize, size_t ysize)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_int clxsize = xsize;
	cl_int clysize = ysize;

	cl_kernel kernel = ocl.kernel[KERNEL_DOMASK];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mask.r);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mask.g);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mask.b);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&mask_dc.r);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&mask_dc.g);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&mask_dc.b);
	clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&clxsize);
	clSetKernelArg(kernel, 7, sizeof(cl_int), (void*)&clysize);

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
}

void clMaskEx(ocl_channels rgb, ocl_channels rgb2,
	size_t xsize, size_t ysize,
	ocl_channels mask/*out*/, ocl_channels mask_dc/*out*/)
{
    clDiffPrecomputeEx(rgb, rgb2, xsize, ysize, mask);
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
        clScaleImageEx(mask.ch[i], xsize * ysize, kGlobalScale * kGlobalScale, mask.ch[i]);
        clScaleImageEx(mask_dc.ch[i], xsize * ysize, kGlobalScale * kGlobalScale, mask_dc.ch[i]);
    }
}

void clCombineChannelsEx(
	ocl_channels mask, 
	ocl_channels mask_dc, 
	cl_mem block_diff_dc, 
	cl_mem block_diff_ac, 
	cl_mem edge_detector_map, 
	size_t xsize, size_t ysize, 
	size_t step, 
	size_t res_xsize,
	cl_mem result/*out*/)
{
	cl_int err = CL_SUCCESS;
	ocl_args_d_t &ocl = getOcl();

	cl_int clxsize = xsize;
	cl_int clysize = ysize;
	cl_int clstep = step;
	cl_int clres_xsize = res_xsize;

	cl_kernel kernel = ocl.kernel[KERNEL_COMBINECHANNELS];
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mask.r);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mask.g);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mask.b);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&mask_dc.r);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&mask_dc.g);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&mask_dc.b);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&block_diff_dc);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&block_diff_ac);
	clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&edge_detector_map);
	clSetKernelArg(kernel, 9, sizeof(cl_int), (void*)&clxsize);
	clSetKernelArg(kernel, 10, sizeof(cl_int), (void*)&clysize);
	clSetKernelArg(kernel, 11, sizeof(cl_int), (void*)&clstep);
	clSetKernelArg(kernel, 12, sizeof(cl_int), (void*)&clres_xsize);
	clSetKernelArg(kernel, 13, sizeof(cl_mem), (void*)&result);

	size_t globalWorkSize[2] = { xsize / step, ysize /step };
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

// strong todo
void clCalculateDiffmapEx(cl_mem result/*in,out*/, size_t xsize, size_t ysize, int step)
{
/*
    int s2 = (8 - step) / 2;
    {
        // Upsample and take square root.
        std::vector<float> diffmap_out(xsize * ysize);
        const size_t res_xsize = (xsize + step - 1) / step;
        for (size_t res_y = 0; res_y + 8 - step < ysize; res_y += step) {
            for (size_t res_x = 0; res_x + 8 - step < xsize; res_x += step) {
                size_t res_ix = (res_y * res_xsize + res_x) / step;
                float orig_val = (*diffmap)[res_ix];
                constexpr float kInitialSlope = 100;
                // TODO(b/29974893): Until that is fixed do not call sqrt on very small
                // numbers.
                double val = orig_val < (1.0 / (kInitialSlope * kInitialSlope))
                    ? kInitialSlope * orig_val
                    : std::sqrt(orig_val);
                for (size_t off_y = 0; off_y < step; ++off_y) {
                    for (size_t off_x = 0; off_x < step; ++off_x) {
                        diffmap_out[(res_y + off_y + s2) * xsize +
                            res_x + off_x + s2] = val;
                    }
                }
            }
        }
        *diffmap = diffmap_out;
    }
*/
    static const double kSigma = 8.8510880283;
    static const double mul1 = 24.8235314874;
    static const double scale = 1.0 / (1.0 + mul1);
    const int s = 8 - step;
    const int s2 = (8 - step) / 2;

    cl_mem blurred;
/*
    for (size_t y = 0; y < ysize - s; ++y) {
    for (size_t x = 0; x < xsize - s; ++x) {
    blurred[y * (xsize - s) + x] = (*diffmap)[(y + s2) * xsize + x + s2];
    }
    }
*/
    static const double border_ratio = 0.03027655136;
    clBlurEx(blurred, xsize - s, ysize - s, kSigma, border_ratio);
/*
    for (size_t y = 0; y < ysize - s; ++y) {
    for (size_t x = 0; x < xsize - s; ++x) {
    (*diffmap)[(y + s2) * xsize + x + s2]
    += static_cast<float>(mul1) * blurred[y * (xsize - s) + x];
    }
    }
*/
    clScaleImageEx(result, xsize * ysize, scale, result);
}

void clDiffmapOpsinDynamicsImage(const float* r, const float* g, const float* b, 
								 float* r2, float* g2, float* b2, 
								 size_t xsize, size_t ysize,
								 size_t step,
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

	cl_mem edge_detector_map = ocl.allocMem(3 * xsize * ysize * sizeof(float));
	cl_mem block_diff_dc = ocl.allocMem(3 * xsize * ysize * sizeof(float));
	cl_mem block_diff_ac = ocl.allocMem(3 * xsize * ysize * sizeof(float));

	ocl_channels mask = ocl.allocMemChannels(channel_size);
	ocl_channels mask_dc = ocl.allocMemChannels(channel_size);
	
	size_t res_xsize_; // 成员变量，需要传递
	size_t res_ysize_; // 成员变量，需要传递
	cl_mem mem_result = ocl.allocMem(channel_size);

	clMaskHighIntensityChangeEx(xyb, xyb2, xsize, ysize);

	clEdgeDetectorMapEx(xyb, xyb2, xsize, ysize, step, edge_detector_map);
	clBlockDiffMapEx(xyb, xyb2, xsize, ysize, block_diff_dc, block_diff_ac);
	clEdgeDetectorLowFreqEx(xyb, xyb2, xsize, ysize, block_diff_ac);

	clMaskEx(xyb, xyb2, xsize, ysize, mask, mask_dc);

	size_t xsize_ = 0, ysize_ = 0; // 成员变量，需要传递
	clCombineChannelsEx(mask, mask_dc, block_diff_dc, block_diff_ac, edge_detector_map, xsize_, ysize_, step, res_xsize_, mem_result);

    clCalculateDiffmapEx(mem_result, xsize, ysize, step);


	cl_float *result_r = (cl_float *)clEnqueueMapBuffer(ocl.commandQueue, mem_result, true, CL_MAP_READ, 0, channel_size, 0, NULL, NULL, &err);
	memcpy(result, result_r, channel_size);

	ocl.releaseMemChannels(xyb);
	ocl.releaseMemChannels(xyb2);

	clReleaseMemObject(edge_detector_map);
	clReleaseMemObject(block_diff_dc);
	clReleaseMemObject(block_diff_ac);

	ocl.releaseMemChannels(mask);
	ocl.releaseMemChannels(mask_dc);

	clReleaseMemObject(mem_result);
}
