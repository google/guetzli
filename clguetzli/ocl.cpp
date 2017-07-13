/*
* OpenCL Manager
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*/
#include "ocl.h"
#include <string.h>
#include <vector>

#ifdef __USE_OPENCL__

ocl_args_d_t& getOcl(void)
{
    static bool bInit = false;
    static ocl_args_d_t ocl;

    if (bInit == true) return ocl;

    bInit = true;
    cl_int err = SetupOpenCL(&ocl, CL_DEVICE_TYPE_GPU);
    LOG_CL_RESULT(err);

    char* source = nullptr;
    size_t src_size = 0;
    ReadSourceFromFile("clguetzli/clguetzli.cl", &source, &src_size);

    ocl.program = clCreateProgramWithSource(ocl.context, 1, (const char**)&source, &src_size, &err);

    delete[] source;

    err = clBuildProgram(ocl.program, 1, &ocl.device, "", NULL, NULL);
    LOG_CL_RESULT(err);
    if (CL_BUILD_PROGRAM_FAILURE == err)
    {
        size_t log_size = 0;
        clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(ocl.program, ocl.device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], NULL);

        LogError("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
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

ocl_args_d_t::ocl_args_d_t() :
	context(NULL),
	device(NULL),
	commandQueue(NULL),
	program(NULL),
	platformVersion(OPENCL_VERSION_1_2),
	deviceVersion(OPENCL_VERSION_1_2),
	compilerVersion(OPENCL_VERSION_1_2)
{
	for (int i = 0; i < KERNEL_COUNT; i++)
	{
		kernel[i] = NULL;
	}
}

ocl_args_d_t::~ocl_args_d_t()
{
	cl_int err = CL_SUCCESS;
	for (int i = 0; i < KERNEL_COUNT; i++)
	{
		err = clReleaseKernel(kernel[i]);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}

	if (program)
	{
		err = clReleaseProgram(program);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseProgram returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (commandQueue)
	{
		err = clReleaseCommandQueue(commandQueue);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseCommandQueue returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (device)
	{
		err = clReleaseDevice(device);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseDevice returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (context)
	{
		err = clReleaseContext(context);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseContext returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
}

cl_mem ocl_args_d_t::allocMem(size_t s, const void *init)
{
	cl_int err = 0;
	cl_mem mem = clCreateBuffer(this->context, CL_MEM_READ_WRITE, s, nullptr, &err);
    LOG_CL_RESULT(err);
    if (!mem) return NULL;
    
    // init memory
    if (init)
    {
        err = clEnqueueWriteBuffer(this->commandQueue, mem, CL_FALSE, 0, s, init, 0, NULL, NULL);
        LOG_CL_RESULT(err);
        err = clFinish(this->commandQueue);
        LOG_CL_RESULT(err);
    }
    else
    {
        cl_char cc = 0;
        err = clEnqueueFillBuffer(this->commandQueue, mem, &cc, sizeof(cc), 0, s / sizeof(cc), 0, NULL, NULL);
        LOG_CL_RESULT(err);
        err = clFinish(this->commandQueue);
        LOG_CL_RESULT(err);
    }

	return mem;
}

ocl_channels ocl_args_d_t::allocMemChannels(size_t s, const void *c0, const void *c1, const void *c2)
{
	const void *c[3] = { c0, c1, c2 };

	ocl_channels img;
    for (int i = 0; i < 3; i++)
    {
        img.ch[i] = allocMem(s, c[i]);
    }

	return img;
}

void ocl_args_d_t::releaseMemChannels(ocl_channels &rgb)
{
    for (int i = 0; i < 3; i++)
    {
        clReleaseMemObject(rgb.ch[i]);
        rgb.ch[i] = NULL;
    }
}

const char* TranslateOpenCLError(cl_int errorCode)
{
	switch (errorCode)
	{
	case CL_SUCCESS:                            return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
	case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
	case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
	case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
	case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
	case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
	case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
	case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
	case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
	case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
	case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
																												//    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
																												//    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    

	default:
		return "UNKNOWN ERROR CODE";
	}
}


/*
* Check whether an OpenCL platform is the required platform
* (based on the platform's name)
*/
bool CheckPreferredPlatformMatch(cl_platform_id platform, const char* preferredPlatform)
{
	size_t stringLength = 0;
	cl_int err = CL_SUCCESS;
	bool match = false;

	// In order to read the platform's name, we first read the platform's name string length (param_value is NULL).
	// The value returned in stringLength
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME length returned '%s'.\n", TranslateOpenCLError(err));
		return false;
	}

	// Now, that we know the platform's name string length, we can allocate enough space before read it
	std::vector<char> platformName(stringLength);

	// Read the platform's name string
	// The read value returned in platformName
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, stringLength, &platformName[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get CL_PLATFORM_NAME returned %s.\n", TranslateOpenCLError(err));
		return false;
	}

	// Now check if the platform's name is the required one
	if (strstr(&platformName[0], preferredPlatform) != 0)
	{
		// The checked platform is the one we're looking for
		match = true;
	}

	return match;
}

/*
* Find and return the preferred OpenCL platform
* In case that preferredPlatform is NULL, the ID of the first discovered platform will be returned
*/
cl_platform_id FindOpenCLPlatform(const char* preferredPlatform, cl_device_type deviceType)
{
	cl_uint numPlatforms = 0;
	cl_int err = CL_SUCCESS;

	// Get (in numPlatforms) the number of OpenCL platforms available
	// No platform ID will be return, since platforms is NULL
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get num platforms returned %s.\n", TranslateOpenCLError(err));
		return NULL;
	}
	LogInfo("Number of available platforms: %u\n", numPlatforms);

	if (0 == numPlatforms)
	{
		LogError("Error: No platforms found!\n");
		return NULL;
	}

	std::vector<cl_platform_id> platforms(numPlatforms);

	// Now, obtains a list of numPlatforms OpenCL platforms available
	// The list of platforms available will be returned in platforms
	err = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get platforms returned %s.\n", TranslateOpenCLError(err));
		return NULL;
	}

	// Check if one of the available platform matches the preferred requirements
	for (cl_uint i = 0; i < numPlatforms; i++)
	{
		bool match = true;
		cl_uint numDevices = 0;

		size_t nameLen = 0;
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &nameLen);

		std::vector<char> platformName(nameLen + 1);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, nameLen, &platformName[0], NULL);
		platformName[nameLen] = 0;

		LogError("DeviceName: %s\n", platformName.data());

		if ((NULL != preferredPlatform) && (strlen(preferredPlatform) > 0))
		{
			match = (strstr(&platformName[0], preferredPlatform) != 0);
		}

		// match is true if the platform's name is the required one or don't care (NULL)
		if (match)
		{
			// Obtains the number of deviceType devices available on platform
			// When the function failed we expect numDevices to be zero.
			// We ignore the function return value since a non-zero error code
			// could happen if this platform doesn't support the specified device type.
			err = clGetDeviceIDs(platforms[i], deviceType, 0, NULL, &numDevices);
			if (CL_SUCCESS != err)
			{
				if (CL_DEVICE_TYPE_GPU == deviceType)
				{
					LogError("%s try GPU returned %s.\n", platformName.data(), TranslateOpenCLError(err));
				}
				if (CL_DEVICE_TYPE_CPU == deviceType)
				{
					LogError("%s try CPU returned %s.\n", platformName.data(), TranslateOpenCLError(err));
				}
			}

			if (0 != numDevices)
			{
				// There is at list one device that answer the requirements
				LogError("SelectDevice: %s GPU=%d\n", platformName.data(), deviceType == CL_DEVICE_TYPE_GPU ? 1 : 0);
				return platforms[i];
			}
		}
	}

	return NULL;
}


/*
* This function read the OpenCL platdorm and device versions
* (using clGetxxxInfo API) and stores it in the ocl structure.
* Later it will enable us to support both OpenCL 1.2 and 2.0 platforms and devices
* in the same program.
*/
int GetPlatformAndDeviceVersion(cl_platform_id platformId, ocl_args_d_t *ocl)
{
	cl_int err = CL_SUCCESS;

	// Read the platform's version string length (param_value is NULL).
	// The value returned in stringLength
	size_t stringLength = 0;
	err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Now, that we know the platform's version string length, we can allocate enough space before read it
	std::vector<char> platformVersion(stringLength);

	// Read the platform's version string
	// The read value returned in platformVersion
	err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, stringLength, &platformVersion[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get CL_PLATFORM_VERSION returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	if (strstr(&platformVersion[0], "OpenCL 2.0") != NULL)
	{
		ocl->platformVersion = OPENCL_VERSION_2_0;
	}

	// Read the device's version string length (param_value is NULL).
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Now, that we know the device's version string length, we can allocate enough space before read it
	std::vector<char> deviceVersion(stringLength);

	// Read the device's version string
	// The read value returned in deviceVersion
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, stringLength, &deviceVersion[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	if (strstr(&deviceVersion[0], "OpenCL 2.0") != NULL)
	{
		ocl->deviceVersion = OPENCL_VERSION_2_0;
	}

	// Read the device's OpenCL C version string length (param_value is NULL).
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Now, that we know the device's OpenCL C version string length, we can allocate enough space before read it
	std::vector<char> compilerVersion(stringLength);

	// Read the device's OpenCL C version string
	// The read value returned in compilerVersion
	err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, stringLength, &compilerVersion[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	else if (strstr(&compilerVersion[0], "OpenCL C 2.0") != NULL)
	{
		ocl->compilerVersion = OPENCL_VERSION_2_0;
	}

	return err;
}


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
int SetupOpenCL(ocl_args_d_t *ocl, cl_device_type deviceType)
{
	// The following variable stores return codes for all OpenCL calls.
	cl_int err = CL_SUCCESS;

	// Query for all available OpenCL platforms on the system
	// Here you enumerate all platforms and pick one which name has preferredPlatform as a sub-string
	cl_platform_id platformId = FindOpenCLPlatform(nullptr, deviceType);
	if (NULL == platformId)
	{
		deviceType = CL_DEVICE_TYPE_CPU;
		platformId = FindOpenCLPlatform(nullptr, deviceType);
	}

	if (NULL == platformId)
	{
		LogError("Error: Failed to find OpenCL platform.\n");
		return CL_INVALID_VALUE;
	}

	// Create context with device of specified type.
	// Required device type is passed as function argument deviceType.
	// So you may use this function to create context for any CPU or GPU OpenCL device.
	// The creation is synchronized (pfn_notify is NULL) and NULL user_data
	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformId, 0 };
	ocl->context = clCreateContextFromType(contextProperties, deviceType, NULL, NULL, &err);
	if ((CL_SUCCESS != err) || (NULL == ocl->context))
	{
		LogError("Couldn't create a context, clCreateContextFromType() returned '%s'.\n", TranslateOpenCLError(err));
		return err;
	}

	// Query for OpenCL device which was used for context creation
	err = clGetContextInfo(ocl->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &ocl->device, NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetContextInfo() to get list of devices returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	// Read the OpenCL platform's version and the device OpenCL and OpenCL C versions
	GetPlatformAndDeviceVersion(platformId, ocl);

	// Create command queue.
	// OpenCL kernels are enqueued for execution to a particular device through special objects called command queues.
	// Command queue guarantees some ordering between calls and other OpenCL commands.
	// Here you create a simple in-order OpenCL command queue that doesn't allow execution of two kernels in parallel on a target device.
#ifdef CL_VERSION_2_0
	if (OPENCL_VERSION_2_0 == ocl->deviceVersion)
	{
		const cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
		ocl->commandQueue = clCreateCommandQueueWithProperties(ocl->context, ocl->device, properties, &err);
	}
	else {
		// default behavior: OpenCL 1.2
		cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
		ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
	}
#else
	// default behavior: OpenCL 1.2
	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
#endif
	if (CL_SUCCESS != err)
	{
		LogError("Error: clCreateCommandQueue() returned %s.\n", TranslateOpenCLError(err));
		return err;
	}

	return CL_SUCCESS;
}

#endif