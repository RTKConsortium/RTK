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
#include "rtkConfiguration.h"
#include "rtkCudaCropImageFilter.hcu"

// cuda includes
#include <cuda.h>
#include <cufft.h>

__global__
void
crop_kernel(float *input, float *output, const long3 cropIdx, const uint3 cropDim, const uint3 inputDim, const unsigned int Blocks_Y)
{
  unsigned int blockIdx_z = blockIdx.y / Blocks_Y;
  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

  if (i >= cropDim.x || j >= cropDim.y || k >= cropDim.z)
    return;

  unsigned long int out_idx = i + j*cropDim.x + k*cropDim.y*cropDim.x;
  unsigned long int in_idx = (cropIdx.x + i) + (cropIdx.y + j)*inputDim.x +  (cropIdx.z + k)*inputDim.y*inputDim.x;

  output[out_idx] = input[in_idx];
}

void
CUDA_crop(const long3 &cropIndex,
          const uint3 &cropSize,
          const uint3 &inputSize,
          float *input,
          float *output)
{
  // Thread Block Dimensions
  unsigned int tBlock_x = 4;
  unsigned int tBlock_y = 4;
  unsigned int tBlock_z = 4;
  unsigned int blocksInX = ( (cropSize.x - 1) / tBlock_x ) + 1;
  unsigned int blocksInY = ( (cropSize.y - 1) / tBlock_y ) + 1;
  unsigned int blocksInZ = ( (cropSize.z - 1) / tBlock_z ) + 1;
  dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

  // Call to kernel
  crop_kernel <<< dimGrid, dimBlock >>> ( input,
                                          output,
                                          cropIndex,
                                          cropSize,
                                          inputSize,
                                          blocksInY );
  CUDA_CHECK_ERROR;

  // Release memory
  //cudaFree(deviceVolume);
}
