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

std::vector<int>
GetListOfCudaDevices()
{
  std::vector<int>      deviceList;
  int                   deviceCount;
  struct cudaDeviceProp properties;
  if (cudaGetDeviceCount(&deviceCount) == cudaSuccess)
  {
    for (int device = 0; device < deviceCount; ++device)
    {
      cudaGetDeviceProperties(&properties, device);
      if (properties.major != 9999) /* 9999 means emulation only */
        deviceList.push_back(device);
    }
  }
  if (deviceList.size() < 1)
    itkGenericExceptionMacro(<< "No CUDA device available");

  return deviceList;
}

std::pair<int, int>
GetCudaComputeCapability(int device)
{
  struct cudaDeviceProp properties;
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    itkGenericExceptionMacro(<< "Invalid CUDA device");
  return std::make_pair(properties.major, properties.minor);
}

size_t
GetFreeGPUGlobalMemory(int device)
{
  // The return result of cuda utility methods are stored in a CUresult
  CUresult result;

  // create cuda context
  CUcontext cudaContext;
  result = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, device);
  if (result != CUDA_SUCCESS)
  {
    itkGenericExceptionMacro(<< "Could not create context on this CUDA device");
  }

  // get the amount of free memory on the graphics card
  size_t free;
  size_t total;
  result = cuMemGetInfo(&free, &total);
  if (result != CUDA_SUCCESS)
  {
    itkGenericExceptionMacro(<< "Could not obtain information on free memory on this CUDA device");
  }

  cuCtxDestroy_v2(cudaContext);

  return free;
}

__host__ void
prepareTextureObject(int                   size[3],
                     float *               dev_ptr,
                     cudaArray **&         componentArrays,
                     unsigned int          nComponents,
                     cudaTextureObject_t * tex,
                     bool                  isProjections)
{
  // Create CUBLAS context
  cublasHandle_t handle;
  cublasCreate(&handle);

  // create texture object
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  for (int component = 0; component < 3; component++)
  {
    if (isProjections)
      texDesc.addressMode[component] = cudaAddressModeBorder;
    else
      texDesc.addressMode[component] = cudaAddressModeClamp;
  }
  texDesc.filterMode = cudaFilterModeLinear;

  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaExtent                   volExtent = make_cudaExtent(size[0], size[1], size[2]);

  // Allocate an intermediate memory space to extract the components of the input volume
  float * singleComponent;
  int     numel = size[0] * size[1] * size[2];
  cudaMalloc(&singleComponent, numel * sizeof(float));
  float one = 1.0;

  // Copy image data to arrays. The tricky part is the make_cudaPitchedPtr.
  // The best way to understand it is to read
  // http://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
  for (unsigned int component = 0; component < nComponents; component++)
  {
    // Reset the intermediate memory
    cudaMemset((void *)singleComponent, 0, numel * sizeof(float));

    // Fill it with the current component
    float * pComponent = dev_ptr + component;
    cublasSaxpy(handle, numel, &one, pComponent, nComponents, singleComponent, 1);

    // Allocate the cudaArray. Projections use layered arrays, volumes use default 3D arrays
    if (isProjections)
      cudaMalloc3DArray((cudaArray **)&componentArrays[component], &channelDesc, volExtent, cudaArrayLayered);
    else
      cudaMalloc3DArray((cudaArray **)&componentArrays[component], &channelDesc, volExtent);

    // Fill it with the current singleComponent
    cudaMemcpy3DParms CopyParams = cudaMemcpy3DParms();
    CopyParams.srcPtr = make_cudaPitchedPtr(singleComponent, size[0] * sizeof(float), size[0], size[1]);
    CopyParams.dstArray = (cudaArray *)componentArrays[component];
    CopyParams.extent = volExtent;
    CopyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&CopyParams);
    CUDA_CHECK_ERROR;

    // Fill in the texture object with all this information
    resDesc.res.array.array = componentArrays[component];
    cudaCreateTextureObject(&tex[component], &resDesc, &texDesc, NULL);
    CUDA_CHECK_ERROR;
  }

  // Intermediate memory is no longer needed
  cudaFree(singleComponent);

  // Destroy CUBLAS context
  cublasDestroy(handle);
}
