/*=========================================================================
 *
 *  Copyright NumFOCUS
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
#ifndef itkCudaUtil_h
#define itkCudaUtil_h

#include <cstring>
#include <cstdlib>
#include <cstdio>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <itkVector.h>
#include <itkMacro.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "itkCudaWin32Header.h"

namespace itk
{

/** Get the local block size based on the desired Image Dimension
 * currently set as follows:
 * Cuda workgroup (block) size for 1/2/3D - needs to be tuned based on the Cuda architecture
 * 1D : 256
 * 2D : 16x16 = 256
 * 3D : 4x4x4 = 64
 */
int
CudaGetLocalBlockSize(unsigned int ImageDim);

std::pair<int, int>
GetCudaComputeCapability(int device);

/** Get the devices that are available */
int
CudaGetAvailableDevices(std::vector<cudaDeviceProp> & devices);

/** Get the device that has the maximum FLOPS in the current context */
int
CudaGetMaxFlopsDev();

/** Print device name and info */
void
CudaPrintDeviceInfo(int device, bool verbose = false);

/** Find the Cuda platform that matches the "name" */
int
CudaSelectPlatform(const char * name);

/** Check Cuda error */
void ITKCudaCommon_EXPORT
     CudaCheckError(cudaError_t error, const char * filename = "", int lineno = 0, const char * location = "");

void ITKCudaCommon_EXPORT
     CudaCheckError(CUresult error, const char * filename = "", int lineno = 0, const char * location = "");

/** Check if Cuda-enabled Cuda is present. */
bool
IsCudaAvailable();

#define CUDA_CHECK(_err_) CudaCheckError(_err_, __FILE__, __LINE__, ITK_LOCATION);
} // namespace itk

#endif
