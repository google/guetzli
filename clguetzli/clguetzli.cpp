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