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
#include "rtkCudaLaplacianImageFilter.hcu"
#include "rtkCudaFirstOrderKernels.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>

void
CUDA_laplacian( int size[3],
                float spacing[3],
                float* dev_in,
                float* dev_out)
{
  int3 dev_Size = make_int3(size[0], size[1], size[2]);
  float3 dev_Spacing = make_float3(spacing[0], spacing[1], spacing[2]);

  // Reset output volume
  long int memorySizeOutput = size[0] * size[1] * size[2] * sizeof(float);
  cudaMemset((void *)dev_out, 0, memorySizeOutput );

  // Initialize volumes to store the gradient components
  float * dev_grad_x;
  float * dev_grad_y;
  float * dev_grad_z;
  cudaMalloc( (void**)&dev_grad_x, memorySizeOutput);
  cudaMalloc( (void**)&dev_grad_y, memorySizeOutput);
  cudaMalloc( (void**)&dev_grad_z, memorySizeOutput);
  cudaMemset(dev_grad_x, 0, memorySizeOutput);
  cudaMemset(dev_grad_y, 0, memorySizeOutput);
  cudaMemset(dev_grad_z, 0, memorySizeOutput);

  // Thread Block Dimensions
  dim3 dimBlock = dim3(16, 4, 4);

  int blocksInX = iDivUp(size[0], dimBlock.x);
  int blocksInY = iDivUp(size[1], dimBlock.y);
  int blocksInZ = iDivUp(size[2], dimBlock.z);

  dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);

  gradient_kernel <<< dimGrid, dimBlock >>> ( dev_in, dev_grad_x, dev_grad_y, dev_grad_z, dev_Size, dev_Spacing);
  CUDA_CHECK_ERROR;

  divergence_kernel <<< dimGrid, dimBlock >>> ( dev_grad_x, dev_grad_y, dev_grad_z, dev_out, dev_Size, dev_Spacing);
  CUDA_CHECK_ERROR;

  // Cleanup
  cudaFree (dev_grad_x);
  cudaFree (dev_grad_y);
  cudaFree (dev_grad_z);
  CUDA_CHECK_ERROR;
}
