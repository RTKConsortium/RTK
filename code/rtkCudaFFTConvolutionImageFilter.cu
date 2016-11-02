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
#include "rtkCudaUtilities.hcu"
#include "rtkCudaFFTConvolutionImageFilter.hcu"

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
#if (CUDA_VERSION<8000)
  result = cufftSetCompatibilityMode(fftFwd, CUFFT_COMPATIBILITY_FFTW_ALL);
  CUFFT_CHECK_ERROR(result);
#endif
  result = cufftExecR2C(fftFwd, deviceProjection, deviceProjectionFFT);
  CUFFT_CHECK_ERROR(result);
  cufftDestroy(fftFwd);

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
#if (CUDA_VERSION<8000)
  result = cufftSetCompatibilityMode(fftInv, CUFFT_COMPATIBILITY_FFTW_ALL);
  CUFFT_CHECK_ERROR(result);
#endif
  result = cufftExecC2R(fftInv, deviceProjectionFFT, deviceProjection);
  CUFFT_CHECK_ERROR(result);

  // Release memory
  cufftDestroy(fftInv);
  cudaFree(deviceProjectionFFT);
}

__global__
void
padding_kernel(float *input,
               float *output,
               const int3 paddingIdx,
               const uint3 paddingDim,
               const uint3 inputDim,
               const unsigned int Blocks_Y,
               float *truncationWeights,
               size_t sizeWeights)
{
  unsigned int blockIdx_z = blockIdx.y / Blocks_Y;
  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
  int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
  int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

  if (i >= paddingDim.x || j >= paddingDim.y || k >= paddingDim.z)
    return;

  unsigned long int out_idx = i + (j + k*paddingDim.y) * paddingDim.x;
  i -= paddingIdx.x;
  j -= paddingIdx.y;
  k -= paddingIdx.z;

  // out of input y/z dimensions
  if(j<0 || j>=inputDim.y || k<0 || k>=inputDim.z)
    output[out_idx] = 0.0f;
  // central part in CPU code
  else if (i>=0 && i<inputDim.x)
    output[out_idx] = input[i + (j + k*inputDim.y) * inputDim.x];
  // left mirroring (equation 3a in [Ohnesorge et al, Med Phys, 2000])
  else if (i<0 && -i<sizeWeights)
    {
    int begRow = (j + k*inputDim.y) * inputDim.x;
    output[out_idx] = (2*input[begRow+1]-input[-i + begRow]) * truncationWeights[-i];
    }
  // right mirroring (equation 3b in [Ohnesorge et al, Med Phys, 2000])
  else if ((i>=inputDim.x) && (i-inputDim.x+1)<sizeWeights)
    {
    unsigned int borderDist = i-inputDim.x+1;
    int endRow = inputDim.x-1 + (j + k*inputDim.y) * inputDim.x;
    output[out_idx] = (2*input[endRow]-input[endRow-borderDist]) * truncationWeights[borderDist];
    }
  // zero padding
  else
    output[out_idx] = 0.0f;
}

void
CUDA_padding(const int3 &paddingIndex,
             const uint3 &paddingSize,
             const uint3 &inputSize,
             float *deviceVolume,
             float *devicePaddedVolume,
             const std::vector<float> &mirrorWeights)
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

  // Transfer weights if required
  float *weights_d = NULL;
  if( mirrorWeights.size() )
    {
    size_t size = mirrorWeights.size() * sizeof(float);
    cudaMalloc((void **) &weights_d, size);
    cudaMemcpy(weights_d, &(mirrorWeights[0]), size, cudaMemcpyHostToDevice);
    }

  // Call to kernel
  padding_kernel <<< dimGrid, dimBlock >>> ( deviceVolume,
                                             devicePaddedVolume,
                                             paddingIndex,
                                             paddingSize,
                                             inputSize,
                                             blocksInY,
                                             weights_d,
                                             mirrorWeights.size() );
  CUDA_CHECK_ERROR;
  // Release memory
  cudaFree(weights_d);
}
