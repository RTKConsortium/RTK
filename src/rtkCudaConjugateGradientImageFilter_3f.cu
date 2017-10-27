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
#include "rtkCudaConjugateGradientImageFilter_3f.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// TEXTURES AND CONSTANTS //

__constant__ int3 c_Size;

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_


__global__
void
subtract_3f(float *in1, float* in2, float* out)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= c_Size.x || j >= c_Size.y || k >= c_Size.z)
      return;

  long int id = (k * c_Size.y + j) * c_Size.x + i;

  out[id] = in1[id] - in2[id];
}

__global__
void
scale_then_add_3f(float *in1, float* in2, float scalar)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= c_Size.x || j >= c_Size.y || k >= c_Size.z)
      return;

  long int id = (k * c_Size.y + j) * c_Size.x + i;

  in1[id] *= scalar;
  in1[id] += in2[id];
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

void
CUDA_copy_3f(int size[3],
              float* in,
              float* out)
{
  int3 dev_Size = make_int3(size[0], size[1], size[2]);
  cudaMemcpyToSymbol(c_Size, &dev_Size, sizeof(int3));

  // Copy input volume to output
  long int memorySizeOutput = size[0] * size[1] * size[2] * sizeof(float);
  cudaMemcpy(out, in, memorySizeOutput, cudaMemcpyDeviceToDevice);
}

void
CUDA_subtract_3f(int size[3],
                 float* in1,
                 float* in2,
                 float* out)
{
  int3 dev_Size = make_int3(size[0], size[1], size[2]);
  cudaMemcpyToSymbol(c_Size, &dev_Size, sizeof(int3));

  // Reset output volume
  long int memorySizeOutput = size[0] * size[1] * size[2] * sizeof(float);
  cudaMemset((void *)out, 0, memorySizeOutput );

  // Thread Block Dimensions
  dim3 dimBlock = dim3(8, 8, 8);

  int blocksInX = iDivUp(size[0], dimBlock.x);
  int blocksInY = iDivUp(size[1], dimBlock.y);
  int blocksInZ = iDivUp(size[2], dimBlock.z);

  dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);

  subtract_3f <<< dimGrid, dimBlock >>> ( in1, in2, out);

  CUDA_CHECK_ERROR;
}

void
CUDA_conjugate_gradient_3f(int size[3],
                           float* Xk,
                           float* Rk,
                           float* Pk,
                           float* APk)
{
  cublasHandle_t  handle;
  cublasCreate(&handle);

  int3 dev_Size = make_int3(size[0], size[1], size[2]);
  cudaMemcpyToSymbol(c_Size, &dev_Size, sizeof(int3));

  // Thread Block Dimensions
  dim3 dimBlock = dim3(8, 8, 8);

  int blocksInX = iDivUp(size[0], dimBlock.x);
  int blocksInY = iDivUp(size[1], dimBlock.y);
  int blocksInZ = iDivUp(size[2], dimBlock.z);

  dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);

  int numel = size[0] * size[1] * size[2];

  // Compute Rk_square = sum(Rk(:).^2) by cublas
  float Rk_square = 0;
  cublasSdot (handle, numel, Rk, 1, Rk, 1, &Rk_square);

  // Compute alpha_k = Rk_square / sum(Pk(:) .* APk(:))
  float Pk_APk = 0;
  cublasSdot (handle, numel, Pk, 1, APk, 1, &Pk_APk);

  const float alpha_k = Rk_square / Pk_APk;
  const float minus_alpha_k = -alpha_k;

  // Compute Xk+1 = Xk + alpha_k * Pk
  cublasSaxpy(handle, numel, &alpha_k, Pk, 1, Xk, 1);

   // Compute Rk+1 = Rk - alpha_k * APk
  cublasSaxpy(handle, numel, &minus_alpha_k, APk, 1, Rk, 1);

  // Compute beta_k = sum(Rk+1(:).^2) / Rk_square
  float Rkplusone_square = 0;
  cublasSdot (handle, numel, Rk, 1, Rk, 1, &Rkplusone_square);

  float beta_k = Rkplusone_square / Rk_square;

  // Compute Pk+1 = Rk+1 + beta_k * Pk
  scale_then_add_3f <<< dimGrid, dimBlock >>> ( Pk, Rk, beta_k);

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}
