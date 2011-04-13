#include "itkCudaUtilities.hcu"

std::vector<int> GetListOfCudaDevices()
{
  std::vector<int>      deviceList;
  int                   deviceCount;
  struct cudaDeviceProp properties;
  cudaError_t           cudaResultCode = cudaGetDeviceCount(&deviceCount);
  if (cudaResultCode == cudaSuccess)
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
