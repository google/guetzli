#include "clguetzli.h"
#include "ocl.h"

void clMinSquareVal(size_t square_size, size_t offset,
	size_t xsize, size_t ysize,
	float *values)
{
	cl_int err = CL_SUCCESS;

	ocl_args_d_t ocl;
	SetupOpenCL(&ocl, CL_DEVICE_TYPE_GPU);

	cl_uint optimizedSize = ((sizeof(cl_float) * xsize * ysize - 1) / 64 + 1) * 64;
	cl_float* inputA = (cl_float*)_aligned_malloc(optimizedSize, 4096);
	cl_float* outputC = (cl_float*)_aligned_malloc(optimizedSize, 4096);

	memcpy(inputA, values, sizeof(cl_float) * xsize * ysize);

	ocl.srcA = clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * xsize * ysize, inputA, &err);
	ocl.dstMem = clCreateBuffer(ocl.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * xsize * ysize, outputC, &err);

	char* source = nullptr;
	size_t src_size = 0;
	ReadSourceFromFile("clguetzli\\clguetzli.cl", &source, &src_size);

	ocl.program = clCreateProgramWithSource(ocl.context, 1, (const char**)&source, &src_size, &err);

	delete[] source;

	err = clBuildProgram(ocl.program, 1, &ocl.device, "", NULL, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));
	}
	ocl.kernel = clCreateKernel(ocl.program, "MinSquareVal", &err);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));
	}

	cl_int cloffset = offset;
	cl_int clsquare_size = square_size;

	clSetKernelArg(ocl.kernel, 0, sizeof(cl_mem), (void*)&ocl.srcA);
	clSetKernelArg(ocl.kernel, 1, sizeof(cl_mem), (void*)&ocl.dstMem);
	clSetKernelArg(ocl.kernel, 2, sizeof(cl_int), (void*)&cloffset);
	clSetKernelArg(ocl.kernel, 3, sizeof(cl_int), (void*)&clsquare_size);

	size_t globalWorkSize[2] = { xsize, ysize };
	err = clEnqueueNDRangeKernel(ocl.commandQueue, ocl.kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
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

	_aligned_free(inputA);
	_aligned_free(outputC);
}