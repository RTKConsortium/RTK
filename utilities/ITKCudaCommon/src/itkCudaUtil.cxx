/*=========================================================================
 *
 *  Copyright Insight Software Consortium
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
#include "itkCudaUtil.h"
#include <assert.h>
#include <iostream>
#include <algorithm>

namespace itk
{
//
// Get the block size based on the desired image dimension
//
int CudaGetLocalBlockSize(unsigned int ImageDim)
{
  /**
   * Cuda thread block size for 1/2/3D - needs to be tuned based on the Cuda architecture
   * 1D : 256
   * 2D : 16x16 = 256
   * 3D : 4x4x4 = 64
   */
  int CUDA_BLOCK_SIZE[3] = { 256, 16, 4 /*8*/ };


  if (ImageDim > 3)
  {
    itkGenericExceptionMacro("Only ImageDimensions up to 3 are supported");
  }
  return CUDA_BLOCK_SIZE[ImageDim-1];
}

//
// Get the devices that are available.
//
int CudaGetAvailableDevices(std::vector<cudaDeviceProp> &devices)
{
  int numAvailableDevices = 0;
  cudaGetDeviceCount(&numAvailableDevices);

  if (numAvailableDevices == 0)
    {
    return 0;
    }

  devices.resize(numAvailableDevices);

  for (int i = 0; i < numAvailableDevices; ++i)
    {
    cudaGetDeviceProperties(&devices[i], i);
    }

  return numAvailableDevices;
}

//
// Get the device that has the maximum FLOPS
//
int CudaGetMaxFlopsDev()
{
  std::vector<cudaDeviceProp> devices;
  int numAvailableDevices = CudaGetAvailableDevices(devices);
  if (numAvailableDevices == 0)
    {

    return -1;
    }
  int max_flops = 0;
  int max_flops_device = 0;
  for (int i = 0; i < numAvailableDevices; ++i)
    {
    int flops = devices[i].multiProcessorCount * devices[i].clockRate;
    if (flops > max_flops)
      {
      max_flops = flops;
      max_flops_device = i;
      }
    }

  return max_flops_device;
}


std::pair<int, int> GetCudaComputeCapability(int device)
{
  struct cudaDeviceProp properties;
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    {
    itkGenericExceptionMacro(<< "Unvalid CUDA device");
    }
  return std::make_pair(properties.major, properties.minor);
}

//
// Print device name & info
//
void CudaPrintDeviceInfo(int device, bool verbose)
{
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess)
    {
    std::cout << "Cuda Error : no device found!" << std::endl;
    return;
    }

  std::cout << prop.name << std::endl;
  std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "Clockrate: " << prop.clockRate << std::endl;
  std::cout << "Global memory: " << prop.totalGlobalMem << std::endl;
  std::cout << "Constant memory: " << prop.totalConstMem << std::endl;
  std::cout << "Number of Multi Processors: " << prop.multiProcessorCount << std::endl;
  std::cout << "Maximum Thread Dim: { " << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << " }" << std::endl;
  std::cout << "Maximum Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "Maximum Grid Size: { " << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << " }" << std::endl;

  if (verbose)
  {
    /*cl_uint mem_align;
    err = clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(mem_align), &mem_align, NULL);
    std::cout << "Alignment in bits of the base address : " << mem_align << std::endl;
    prop.ext
    cl_uint min_align;
    err = clGetDeviceInfo(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(min_align), &min_align, NULL);
    std::cout << "Smallest alignment in bytes for any data type : " << min_align << std::endl;

    char device_extensions[1024];
    err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(device_extensions), &device_extensions, NULL);
    printf("%s\n", device_extensions);*/
  }
}

//
// Find the Cuda platform that matches the "name"
//
int CudaSelectPlatform(const char* name)
{
  int numAvailableDevices = 0;
  std::vector<cudaDeviceProp> devices;
  numAvailableDevices = CudaGetAvailableDevices(devices);
  if (numAvailableDevices == 0)
    {
    std::cout << "Cuda Error : no device found!" << std::endl;
    return -1;
    }

  for (int i = 0; i < numAvailableDevices; ++i)
    {
    if (!strcmp(devices[i].name, name))
      {
      return i;
      }
    }

  return -1;
}

void CudaCheckError(cudaError_t error, const char* filename, int lineno, const char* location)
{
  if (error != cudaSuccess)
    {
    // print error message
    std::ostringstream errorMsg;
    errorMsg << "Cuda Error : " << cudaGetErrorString(error) << std::endl;
    std::cerr << filename << ":" << lineno << " @ " << location << " : " << errorMsg.str() << std::endl;
    ::itk::ExceptionObject e_(filename, lineno, errorMsg.str().c_str(), location);
    throw e_;
    }
}


void CudaCheckError(CUresult error, const char* filename, int lineno, const char* location)
{
  if (error != CUDA_SUCCESS)
    {
    // print error message
    std::ostringstream errorMsg;
    errorMsg << "Cuda Error #" << static_cast<int>(error) << std::endl;
    std::cerr << errorMsg.str() << std::endl;
    ::itk::ExceptionObject e_(filename, lineno, errorMsg.str().c_str(), location);
    throw e_;
    }
}


/** Check if OpenCL-enabled Cuda is present. */
bool IsCudaAvailable()
{
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);

  return count >= 1;
}

std::string GetTypename(const std::type_info& intype)
{
  std::string typestr;
  if (intype == typeid (unsigned char) ||
       intype == typeid (itk::Vector< unsigned char, 2 >) ||
       intype == typeid (itk::Vector< unsigned char, 3 >))
    {
    typestr = "uc";
    }
  else if (intype == typeid (char) ||
            intype == typeid (itk::Vector< char, 2 >) ||
            intype == typeid (itk::Vector< char, 3 >))
    {
    typestr = "c";
    }
  else if (intype == typeid (short) ||
            intype == typeid (itk::Vector< short, 2 >) ||
            intype == typeid (itk::Vector< short, 3 >))
    {
    typestr = "s";
    }
  else if (intype == typeid (int) ||
            intype == typeid (itk::Vector< int, 2 >) ||
            intype == typeid (itk::Vector< int, 3 >))
    {
    typestr = "i";
    }
  else if (intype == typeid (unsigned int) ||
            intype == typeid (itk::Vector< unsigned int, 2 >) ||
            intype == typeid (itk::Vector< unsigned int, 3 >))
    {
    typestr = "ui";
    }
  else if (intype == typeid (long) ||
            intype == typeid (itk::Vector< long, 2 >) ||
            intype == typeid (itk::Vector< long, 3 >))
    {
    typestr = "l";
    }
  else if (intype == typeid (unsigned long) ||
            intype == typeid (itk::Vector< unsigned long, 2 >) ||
            intype == typeid (itk::Vector< unsigned long, 3 >))
    {
    typestr = "ul";
    }
  else if (intype == typeid (long long) ||
            intype == typeid (itk::Vector< long long, 2 >) ||
            intype == typeid (itk::Vector< long long, 3 >))
    {
    typestr = "ll";
    }
  else if (intype == typeid (float) ||
            intype == typeid (itk::Vector< float, 2 >) ||
            intype == typeid (itk::Vector< float, 3 >))
    {
    typestr = "f";
    }
  else if (intype == typeid (double) ||
            intype == typeid (itk::Vector< double, 2 >) ||
            intype == typeid (itk::Vector< double, 3 >))
    {
    typestr = "d";
    }
  else
    {
      itkGenericExceptionMacro("Unknown type: " << intype.name());
    }
  return typestr;
}

/** Get Typename in String if a valid type */
bool GetValidTypename(const std::type_info& intype, const std::vector<std::string>& validtypes, std::string& retTypeName)
{
  std::string typestr = GetTypename(intype);
  bool isValid = false;
  std::vector<std::string>::const_iterator validPos;
  validPos = std::find(validtypes.begin(), validtypes.end(), typestr);
  if (validPos != validtypes.end())
    {
      isValid = true;
      retTypeName = *validPos;
    }

  return isValid;
}

/** Get 64-bit pragma */
std::string Get64BitPragma()
{
  std::ostringstream msg;
  //msg << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
  //msg << "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n";
  return msg.str();
}

void GetTypenameInString(const std::type_info& intype, std::ostringstream& ret)
{
  std::string typestr = GetTypename(intype);
  ret << typestr;
  /*if (typestr == "d")
    {
    std::string pragmastr = Get64BitPragma();
    ret << pragmastr;
    }*/
}

int GetPixelDimension(const std::type_info& intype)
{
  if (intype == typeid (unsigned char) ||
       intype == typeid (char) ||
       intype == typeid (short) ||
       intype == typeid (int) ||
       intype == typeid (unsigned int) ||
       intype == typeid (float) ||
       intype == typeid (double))
    {
    return 1;
    }
  else if (intype == typeid (itk::Vector< unsigned char, 2 >) ||
           intype == typeid (itk::Vector< char, 2 >) ||
           intype == typeid (itk::Vector< short, 2 >) ||
           intype == typeid (itk::Vector< int, 2 >) ||
           intype == typeid (itk::Vector< unsigned int, 2 >) ||
           intype == typeid (itk::Vector< float, 2 >) ||
           intype == typeid (itk::Vector< double, 2 >))
    {
    return 2;
    }
  else if (intype == typeid (itk::Vector< unsigned char, 3 >) ||
           intype == typeid (itk::Vector< char, 3 >) ||
           intype == typeid (itk::Vector< short, 3 >) ||
           intype == typeid (itk::Vector< int, 3 >) ||
           intype == typeid (itk::Vector< unsigned int, 3 >) ||
           intype == typeid (itk::Vector< float, 3 >) ||
           intype == typeid (itk::Vector< double, 3 >))
    {
    return 3;
    }
  else
    {
    itkGenericExceptionMacro("Pixeltype is not supported by the filter.");
    }
}

} // end namespace itk
