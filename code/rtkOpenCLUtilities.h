/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __rtkOpenCLUtilities_h
#define __rtkOpenCLUtilities_h

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
