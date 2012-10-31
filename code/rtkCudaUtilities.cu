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

#include "rtkCudaUtilities.hcu"

std::vector<int> GetListOfCudaDevices()
{
  std::vector<int>      deviceList;
  int                   deviceCount;
  struct cudaDeviceProp properties;
  if (cudaGetDeviceCount(&deviceCount) == cudaSuccess)
    {
    for (int device = 0; device < deviceCount; ++device) {
      cudaGetDeviceProperties(&properties, device);
      if (properties.major != 9999)   /* 9999 means emulation only */
        deviceList.push_back(device);
      }
    }
  if(deviceList.size()<1)
    itkGenericExceptionMacro(<< "No CUDA device available");

  return deviceList;
}

std::pair<int,int> GetCudaComputeCapability(int device)
{
  struct cudaDeviceProp properties;
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    itkGenericExceptionMacro(<< "Unvalid CUDA device");
  return std::make_pair(properties.major, properties.minor);
}
