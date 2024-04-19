/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtkCudaUtilities.hcu"
#include <cublas_v2.h>

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
prepareScalarTextureObject(int                          size[3],
                           float *                      dev_ptr,
                           cudaArray *&                 threeDArray,
                           cudaTextureObject_t &        tex,
                           const bool                   isProjections,
                           const bool                   isLinear,
                           const cudaTextureAddressMode texAddressMode)
{
  // create texture object
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;

  cudaTextureDesc texDesc = {};
  texDesc.readMode = cudaReadModeElementType;

  for (int component = 0; component < 3; component++)
    texDesc.addressMode[component] = texAddressMode;
  if (isLinear)
    texDesc.filterMode = cudaFilterModeLinear;
  else
    texDesc.filterMode = cudaFilterModePoint;

  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaExtent                   volExtent = make_cudaExtent(size[0], size[1], size[2]);

  // Allocate an intermediate memory space to extract the components of the input volume
  float * singleComponent;
  int     numel = size[0] * size[1] * size[2];
  cudaMalloc(&singleComponent, numel * sizeof(float));
  CUDA_CHECK_ERROR;

  // Copy image data to arrays. The tricky part is the make_cudaPitchedPtr.
  // The best way to understand it is to read
  // https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api

  // Allocate the cudaArray. Projections use layered arrays, volumes use default 3D arrays
  if (isProjections)
    cudaMalloc3DArray(&threeDArray, &channelDesc, volExtent, cudaArrayLayered);
  else
    cudaMalloc3DArray(&threeDArray, &channelDesc, volExtent);
  CUDA_CHECK_ERROR;

  // Fill it with the current singleComponent
  cudaMemcpy3DParms CopyParams = {};
  CopyParams.srcPtr = make_cudaPitchedPtr(dev_ptr, size[0] * sizeof(float), size[0], size[1]);
  CUDA_CHECK_ERROR;
  CopyParams.dstArray = threeDArray;
  CopyParams.extent = volExtent;
  CopyParams.kind = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&CopyParams);
  CUDA_CHECK_ERROR;

  // Fill in the texture object with all this information
  resDesc.res.array.array = threeDArray;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
  CUDA_CHECK_ERROR;
}

__host__ void
prepareVectorTextureObject(int                                size[3],
                           const float *                      dev_ptr,
                           std::vector<cudaArray *> &         componentArrays,
                           const unsigned int                 nComponents,
                           std::vector<cudaTextureObject_t> & tex,
                           const bool                         isProjections,
                           const cudaTextureAddressMode       texAddressMode)
{
  componentArrays.resize(nComponents);
  tex.resize(nComponents);

  // Create CUBLAS context
  cublasHandle_t handle;
  cublasCreate(&handle);

  // create texture object
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;

  cudaTextureDesc texDesc = {};
  texDesc.readMode = cudaReadModeElementType;
  for (int component = 0; component < 3; component++)
    texDesc.addressMode[component] = texAddressMode;
  texDesc.filterMode = cudaFilterModeLinear;

  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaExtent                   volExtent = make_cudaExtent(size[0], size[1], size[2]);

  // Allocate an intermediate memory space to extract the components of the input volume
  float * singleComponent;
  int     numel = size[0] * size[1] * size[2];
  cudaMalloc(&singleComponent, numel * sizeof(float));
  CUDA_CHECK_ERROR;
  float one = 1.0;

  // Copy image data to arrays. The tricky part is the make_cudaPitchedPtr.
  // The best way to understand it is to read
  // https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
  for (unsigned int component = 0; component < nComponents; component++)
  {
    // Reset the intermediate memory
    cudaMemset((void *)singleComponent, 0, numel * sizeof(float));

    // Fill it with the current component
    const float * pComponent = dev_ptr + component;
    cublasSaxpy(handle, numel, &one, pComponent, nComponents, singleComponent, 1);

    // Allocate the cudaArray. Projections use layered arrays, volumes use default 3D arrays
    if (isProjections)
      cudaMalloc3DArray(&componentArrays[component], &channelDesc, volExtent, cudaArrayLayered);
    else
      cudaMalloc3DArray(&componentArrays[component], &channelDesc, volExtent);
    CUDA_CHECK_ERROR;

    // Fill it with the current singleComponent
    cudaMemcpy3DParms CopyParams = cudaMemcpy3DParms();
    CopyParams.srcPtr = make_cudaPitchedPtr(singleComponent, size[0] * sizeof(float), size[0], size[1]);
    CUDA_CHECK_ERROR;
    CopyParams.dstArray = componentArrays[component];
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

__host__ void
prepareGeometryTextureObject(int                   length,
                             const float *         geometry,
                             float *&              dev_geom,
                             cudaTextureObject_t & tex_geom,
                             const unsigned int    nParam)
{
  // copy geometry matrix to device, bind the matrix to the texture
  cudaMalloc((void **)&dev_geom, length * nParam * sizeof(float));
  CUDA_CHECK_ERROR;
  cudaMemcpy(dev_geom, geometry, length * nParam * sizeof(float), cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR;

  // create texture object
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = dev_geom;
  resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
  resDesc.res.linear.desc.x = 32; // bits per channel
  resDesc.res.linear.sizeInBytes = length * nParam * sizeof(float);

  cudaTextureDesc texDesc = {};
  texDesc.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(&tex_geom, &resDesc, &texDesc, NULL);
  CUDA_CHECK_ERROR;
}
