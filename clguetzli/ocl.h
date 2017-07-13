/*
* OpenCL Manager
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*/
#pragma once

#ifdef __USE_OPENCL__

#include "CL/cl.h"
#include "utils.h"
#include "clguetzli.cl.h"

// Macros for OpenCL versions
#define OPENCL_VERSION_1_2  1.2f
#define OPENCL_VERSION_2_0  2.0f

enum KernelName {
    KERNEL_CONVOLUTION = 0,
    KERNEL_CONVOLUTIONX,
    KERNEL_CONVOLUTIONY,
    KERNEL_SQUARESAMPLE,
    KERNEL_OPSINDYNAMICSIMAGE,
    KERNEL_MASKHIGHINTENSITYCHANGE,
    KERNEL_EDGEDETECTOR,
    KERNEL_BLOCKDIFFMAP,
    KERNEL_EDGEDETECTORLOWFREQ,
    KERNEL_DIFFPRECOMPUTE,
    KERNEL_SCALEIMAGE,
    KERNEL_AVERAGE5X5,
    KERNEL_MINSQUAREVAL,
    KERNEL_DOMASK,
    KERNEL_COMBINECHANNELS,
    KERNEL_UPSAMPLESQUAREROOT,
    KERNEL_REMOVEBORDER,
    KERNEL_ADDBORDER,
    KERNEL_COMPUTEBLOCKZEROINGORDER,
    KERNEL_COUNT,
};

#define LOG_CL_RESULT(e)   if (CL_SUCCESS != (e)) { LogError("Error: %s:%d returned %s.\n", __FUNCTION__, __LINE__, TranslateOpenCLError((e)));}

struct ocl_args_d_t;

const char* TranslateOpenCLError(cl_int errorCode);

int SetupOpenCL(ocl_args_d_t *ocl, cl_device_type deviceType);

ocl_args_d_t& getOcl(void);

struct ocl_args_d_t
{
	ocl_args_d_t();
	~ocl_args_d_t();

	cl_mem allocMem(size_t s, const void *init = NULL);
	ocl_channels allocMemChannels(size_t s, const void *c0 = NULL, const void *c1 = NULL, const void *c2 = NULL);
    void releaseMemChannels(ocl_channels &rgb);

	// Regular OpenCL objects:
	cl_context       context;           // hold the context handler
	cl_device_id     device;            // hold the selected device handler
	cl_command_queue commandQueue;      // hold the commands-queue handler
	cl_program       program;           // hold the program handler
	cl_kernel        kernel[KERNEL_COUNT];            // hold the kernel handler
	float            platformVersion;   // hold the OpenCL platform version (default 1.2)
	float            deviceVersion;     // hold the OpenCL device version (default. 1.2)
	float            compilerVersion;   // hold the device OpenCL C version (default. 1.2)
};

#endif
