#pragma once

#include "CL\cl.h"
#include "utils.h"

// Macros for OpenCL versions
#define OPENCL_VERSION_1_2  1.2f
#define OPENCL_VERSION_2_0  2.0f

struct ocl_args_d_t;

/* This function helps to create informative messages in
* case when OpenCL errors occur. It returns a string
* representation for an OpenCL error code.
* (E.g. "CL_DEVICE_NOT_FOUND" instead of just -1.)
*/
const char* TranslateOpenCLError(cl_int errorCode);

/*
* This function picks/creates necessary OpenCL objects which are needed.
* The objects are:
* OpenCL platform, device, context, and command queue.
*
* All these steps are needed to be performed once in a regular OpenCL application.
* This happens before actual compute kernels calls are performed.
*
* For convenience, in this application you store all those basic OpenCL objects in structure ocl_args_d_t,
* so this function populates fields of this structure, which is passed as parameter ocl.
* Please, consider reviewing the fields before going further.
* The structure definition is right in the beginning of this file.
*/
int SetupOpenCL(ocl_args_d_t *ocl, cl_device_type deviceType);


/* Convenient container for all OpenCL specific objects used in the sample
*
* It consists of two parts:
*   - regular OpenCL objects which are used in almost each normal OpenCL applications
*   - several OpenCL objects that are specific for this particular sample
*
* You collect all these objects in one structure for utility purposes
* only, there is no OpenCL specific here: just to avoid global variables
* and make passing all these arguments in functions easier.
*/

enum KernelName {
	KERNEL_MINSQUAREVAL = 0,
	KERNEL_CONVOLUTION,
	KERNEL_CONVOLUTIONX,
	KERNEL_CONVOLUTIONY,
	KERNEL_DOWNSAMPLE,
	KERNEL_OPSINDYNAMICSIMAGE,
	KERNEL_DOMASK,
	KERNEL_SCALEIMAGE,
	KERNEL_COMBINECHANNELS,
	KERNEL_MASKHIGHINTENSITYCHANGE,
	KERNEL_DIFFPRECOMPUTE,
	KERNEL_UPSAMPLESQUAREROOT,
	KERNEL_CALCULATEDIFFMAPGETBLURRED,
	KERNEL_GETDIFFMAPFROMBLURRED,
	KERNEL_AVERAGEADDIMAGE,
	KERNEL_EDGEDETECTOR,
	KERNEL_BLOCKDIFFMAP,
	KERNEL_EDGEDETECTORLOWFREQ,
	KERNEL_COUNT,
};

typedef union ocl_channels_t
{
    struct
    {
        cl_mem r;
        cl_mem g;
        cl_mem b;
    };

    cl_mem ch[3];
}ocl_channels;

struct ocl_args_d_t
{
	ocl_args_d_t();
	~ocl_args_d_t();

	void* allocA(size_t s);
	void* allocB(size_t s);
	void* allocC(size_t s);

	cl_mem allocMem(size_t s);
	ocl_channels allocMemChannels(size_t s);
    void releaseMemChannels(ocl_channels rgb);

	// Regular OpenCL objects:
	cl_context       context;           // hold the context handler
	cl_device_id     device;            // hold the selected device handler
	cl_command_queue commandQueue;      // hold the commands-queue handler
	cl_program       program;           // hold the program handler
	cl_kernel        kernel[KERNEL_COUNT];            // hold the kernel handler
	float            platformVersion;   // hold the OpenCL platform version (default 1.2)
	float            deviceVersion;     // hold the OpenCL device version (default. 1.2)
	float            compilerVersion;   // hold the device OpenCL C version (default. 1.2)

										// Objects that are specific for algorithm implemented in this sample
	cl_mem           srcA;              // hold first source buffer
	cl_mem           srcB;              // hold second source buffer
	cl_mem           dstMem;            // hold destination buffer

	void*			 inputA;
	size_t		     lenA;

	void*			 inputB;
	size_t			 lenB;

	void*			 outputC;
	size_t			 lenC;
};
