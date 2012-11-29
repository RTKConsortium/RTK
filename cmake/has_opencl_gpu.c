#include <stdio.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main() {
  
  cl_uint numberOfPlatforms;
  if( clGetPlatformIDs (0, NULL, &numberOfPlatforms) )
    {
    return 1; // failure
    }

  std::vector<cl_platform_id> platformList(numberOfPlatforms);
  if( clGetPlatformIDs (numberOfPlatforms, &(platformList[0]), NULL) )
    {
    return 1; // failure
    }

  cl_uint numberOfDevices;
  if(clGetDeviceIDs(platformList[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numberOfDevices) ) != CL_SUCCESS)
    {
    return 1; // failure
    }

  std::vector<cl_device_id> deviceList(numberOfDevices);
  if(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numberOfDevices, &(deviceList[0]), NULL) != CL_SUCCESS)
    {
    return 1; // failure
    }

  cl_bool bImageSupport = false;
  // If found, check if supports image.
  if(numberOfDevices>0)
    if( clGetDeviceInfo (deviceList[0], CL_DEVICE_IMAGE_SUPPORT,
                                         sizeof(cl_bool), &bImageSupport, NULL) )
      {
      return 1; // failure
      }

  // If not a good device, switch to CPU.
  if(!bImageSupport)
    {
    if( clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numberOfDevices) )
      {
      return 1; // failure
      }
    deviceList.resize(numberOfDevices);
    if( clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numberOfDevices, &(deviceList[0]), NULL) )
      {
      return 1; // failure
      }
    }
  printf("%d GPU OpenCL device(s) found\n", numberOfDevices);

  /* don't just return the number of gpus, because other runtime cuda
     errors can also yield non-zero return values */
  if (numberOfDevices > 0)
      return 0; /* success */
  else
      return 1; /* failure */
}
