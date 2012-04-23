#ifndef __rtkOpenCLUtilities_hcu
#define __rtkOpenCLUtilities_hcu

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include <vector>
#include <itkMacro.h>

#define OPENCL_CHECK_ERROR(cmd) \
    { \
    cl_int rc = cmd; \
    if (rc != CL_SUCCESS) \
      itkGenericExceptionMacro(<< "OPENCL ERROR with " \
                               << #cmd \
                               << ". Returned value is " \
                               << rc); \
    }

std::vector<cl_platform_id> GetListOfOpenCLPlatforms();

std::vector<cl_device_id> GetListOfOpenCLDevices(const cl_platform_id platform);

void CreateAndBuildOpenCLProgramFromSourceFile(const std::string fileName, const cl_context &context,
                                               cl_program &program);

#endif
