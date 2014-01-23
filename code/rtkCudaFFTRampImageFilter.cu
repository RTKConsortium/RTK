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

// rtk includes
#include "rtkCudaFFTRampImageFilter.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>
#include <cufft.h>

__global__
void
multiply_kernel(cufftComplex *projFFT, int3 fftDimension, cufftComplex *kernelFFT, unsigned int Blocks_Y)
{
  unsigned int blockIdx_z = blockIdx.y / Blocks_Y;
  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

  if (i >= fftDimension.x || j >= fftDimension.y || k >= fftDimension.z)
    return;

  long int proj_idx = i + (j + k * fftDimension.y ) * fftDimension.x;

  cufftComplex result;
  result.x = projFFT[proj_idx].x * kernelFFT[i].x - projFFT[proj_idx].y * kernelFFT[i].y;
  result.y = projFFT[proj_idx].y * kernelFFT[i].x + projFFT[proj_idx].x * kernelFFT[i].y;
  projFFT[proj_idx] = result;
}

__global__
void
multiply_kernel2D(cufftComplex *projFFT,
                  int3 fftDimension,
                  cufftComplex *kernelFFT,
                  unsigned int Blocks_Y)
{
  unsigned int blockIdx_z = blockIdx.y / Blocks_Y;
  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

  if (i >= fftDimension.x || j >= fftDimension.y || k >= fftDimension.z)
    return;

  long int kernel_idx = i + j * fftDimension.x;
  long int proj_idx = kernel_idx + k * fftDimension.y * fftDimension.x;

  cufftComplex result;
  result.x = projFFT[proj_idx].x * kernelFFT[kernel_idx].x - projFFT[proj_idx].y * kernelFFT[kernel_idx].y;
  result.y = projFFT[proj_idx].y * kernelFFT[kernel_idx].x + projFFT[proj_idx].x * kernelFFT[kernel_idx].y;
  projFFT[proj_idx] = result;
}

void
CUDA_fft_convolution(const int3 &inputDimension,
                     const int2 &kernelDimension,
                     float *deviceProjection,
                     cufftComplex *deviceKernelFFT)
{
  // CUDA device pointers
  cufftComplex *deviceProjectionFFT;
  int3          fftDimension = inputDimension;
  fftDimension.x = inputDimension.x / 2 + 1;

  int memorySizeProjectionFFT = fftDimension.x*fftDimension.y*fftDimension.z*sizeof(cufftComplex);
  cudaMalloc( (void**)&deviceProjectionFFT, memorySizeProjectionFFT);
  CUDA_CHECK_ERROR;

  // 3D FFT
  cufftHandle fftFwd;
  cufftResult result;
  if(fftDimension.z==1)
    result = cufftPlan2d(&fftFwd, inputDimension.y, inputDimension.x, CUFFT_R2C);
  else
    result = cufftPlan3d(&fftFwd, inputDimension.z, inputDimension.y, inputDimension.x, CUFFT_R2C);
  CUFFT_CHECK_ERROR(result);
  result = cufftSetCompatibilityMode(fftFwd, CUFFT_COMPATIBILITY_FFTW_ALL);
  CUFFT_CHECK_ERROR(result);
  result = cufftExecR2C(fftFwd, deviceProjection, deviceProjectionFFT);
  CUFFT_CHECK_ERROR(result);

  // Thread Block Dimensions
  int tBlock_x = 16;
  int tBlock_y = 4;
  int tBlock_z = 4;
  int  blocksInX = (fftDimension.x - 1) / tBlock_x + 1;
  int  blocksInY = (fftDimension.y - 1) / tBlock_y + 1;
  int  blocksInZ = (fftDimension.z - 1) / tBlock_z + 1;
  dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
  if(kernelDimension.y==1)
    multiply_kernel <<< dimGrid, dimBlock >>> ( deviceProjectionFFT,
                                                fftDimension,
                                                deviceKernelFFT,
                                                blocksInY );
  else
    multiply_kernel2D <<< dimGrid, dimBlock >>> ( deviceProjectionFFT,
                                                  fftDimension,
                                                  deviceKernelFFT,
                                                  blocksInY );
  CUDA_CHECK_ERROR;

  // 3D inverse FFT
  cufftHandle fftInv;
  if(fftDimension.z==1)
    result = cufftPlan2d(&fftInv, inputDimension.y, inputDimension.x, CUFFT_C2R);
  else
    result = cufftPlan3d(&fftInv, inputDimension.z, inputDimension.y, inputDimension.x, CUFFT_C2R);
  CUFFT_CHECK_ERROR(result);
  result = cufftSetCompatibilityMode(fftInv, CUFFT_COMPATIBILITY_FFTW_ALL);
  CUFFT_CHECK_ERROR(result);
  result = cufftExecC2R(fftInv, deviceProjectionFFT, deviceProjection);
  CUFFT_CHECK_ERROR(result);

  // Release memory
  cufftDestroy(fftFwd);
  cufftDestroy(fftInv);
  cudaFree(deviceProjectionFFT);
}

// ******************** Padding Section ***************************
__global__
void
padding_kernel(float *input, float *output, const long3 paddingIdx, const uint3 paddingDim, const unsigned int Blocks_Y, float *truncationWeights, unsigned int sizeWeights, float hannY)
{
  unsigned int blockIdx_z = blockIdx.y / Blocks_Y;
  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

  if (i >= paddingDim.x || j >= paddingDim.y || k >= paddingDim.z)
    return;

  unsigned long int out_idx = i + j*paddingDim.x + k*paddingDim.y*paddingDim.x;
  unsigned long int mirror_idx = 0;
  unsigned int condition = 0;
  long int in_idx = -1;

  if(hannY!=0) // hannY filtering ON
  {
    // Middle zone
    if(i>=paddingDim.x*0.25f && i<=paddingDim.x*0.75f)
    {
      if(j>=paddingDim.y*0.5f)
        condition=0;
      else
        in_idx = (i-paddingDim.x*0.25f) + j*paddingDim.x*0.5f + k*paddingDim.y*paddingDim.x*0.25f;
    }
    // Left padded zone
    else if(i<paddingDim.x*0.25f)
    {
      if(j>=paddingDim.y*0.5f)
        condition=0;
      else
      {
        //mirror_idx = (j-paddingDim.y*0.25f)*paddingDim.x*0.5f - (i-(paddingDim.x*0.25f+1)) + k*paddingDim.y*paddingDim.x*0.25f;
        mirror_idx = j*paddingDim.x*0.5f - (i-(paddingDim.x*0.25f+1)) + k*paddingDim.y*paddingDim.x*0.25f;
        condition = (paddingDim.x*0.25f-1) - i;
      }
    }
    // Right padded zone
    else if(i>=paddingDim.x*0.75f)
    {
      if(j>=paddingDim.y*0.5f)
        condition=0;
      else
      {
        //mirror_idx = (j-paddingDim.y*0.25f)*paddingDim.x*0.5f - (i-(paddingDim.x*1.25f-1)) + k*paddingDim.y*paddingDim.x*0.25f;
        mirror_idx = j*paddingDim.x*0.5f - (i-(paddingDim.x*1.25f-1)) + k*paddingDim.y*paddingDim.x*0.25f;
        condition = i - (paddingDim.x*0.75f-1);
      }
    }
  }
  else  // hannY filtering OFF
  {
    // Original image zone
    if(i>=paddingDim.x*0.25f && i<=paddingDim.x*0.75f)
      in_idx = (i-paddingDim.x*0.25f) + j*paddingDim.x*0.5f + k*paddingDim.y*paddingDim.x*0.5f;
    // Left padded zone
    else if(i<paddingDim.x*0.25f)
    {
      if(j<paddingDim.y*0.25f || j>paddingDim.y*0.75f)
        condition=0;
      else
      {
        mirror_idx = j*paddingDim.x*0.5f - (i-(paddingDim.x*0.25f+1)) + k*paddingDim.y*paddingDim.x*0.5f;
        condition = (paddingDim.x*0.25f-1) - i;
      }
    }
    // Right padded zone
    else if(i>=paddingDim.x*0.75f)
    {
      if(j<paddingDim.y*0.25f || j>paddingDim.y*0.75f)
        condition=0;
      else
      {
        mirror_idx = j*paddingDim.x*0.5f - (i-(paddingDim.x*1.25f-1)) + k*paddingDim.y*paddingDim.x*0.5f;
        condition = i - (paddingDim.x*0.75f-1);
      }
    }
  }

  // Copying original image
  if(in_idx!=-1)
    output[out_idx] = input[in_idx];
  // Mirroring left and right zones
  else
  {
    if(condition>sizeWeights)
      output[out_idx] = 0;
    else
      output[out_idx] = truncationWeights[condition]*input[mirror_idx];
  }
}

void
CUDA_padding(const long3 &paddingIndex,
             const uint3 &paddingSize,
             float *input,
             float *output,
             float *truncationWeights,
             unsigned int sizeWeights,
             float hannY)
{
  // Thread Block Dimensions
  unsigned int tBlock_x = 4;
  unsigned int tBlock_y = 4;
  unsigned int tBlock_z = 4;
  unsigned int blocksInX = ( (paddingSize.x - 1) / tBlock_x ) + 1;
  unsigned int blocksInY = ( (paddingSize.y - 1) / tBlock_y ) + 1;
  unsigned int blocksInZ = ( (paddingSize.z - 1) / tBlock_z ) + 1;
  dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

  float *weights_d;   // Pointer to host & device arrays of structure
  int size = sizeWeights*sizeof(float);
  // Allocate weighting array on device
  cudaMalloc((void **) &weights_d, size);
  cudaMemcpy(weights_d, truncationWeights, size, cudaMemcpyHostToDevice);
  // Call to kernel
  padding_kernel <<< dimGrid, dimBlock >>> ( input,
                                             output,
                                             paddingIndex,
                                             paddingSize,
                                             blocksInY,
                                             weights_d,
                                             sizeWeights,
                                             hannY);
  CUDA_CHECK_ERROR;

  // Release memory
  cudaFree(weights_d);
}
