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
#include "rtkCudaInterpolateImageFilter.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>

__global__
void
splat_kernel(float *input, int4 outputSize, float* output, int phase, float weight, unsigned int Blocks_Y)
{
  // CUDA 2.0 does not allow for a 3D grid, which severely
  // limits the manipulation of large 3D arrays of data.  The
  // following code is a hack to bypass this implementation
  // limitation.
  unsigned int blockIdx_z = blockIdx.y / Blocks_Y;
  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

  if (i >= outputSize.x || j >= outputSize.y || k >= outputSize.z || phase >= outputSize.w)
      return;

  long int output_idx = ((phase * outputSize.z + k) * outputSize.y + j) * outputSize.x + i;
  long int input_idx = (k * outputSize.y + j) * outputSize.x + i;

  output[output_idx] += input[input_idx] * weight;
}

__global__
void
splat_kernel_3Dgrid(float *input, int4 outputSize, float* output, int phase, float weight)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= outputSize.x || j >= outputSize.y || k >= outputSize.z || phase >= outputSize.w)
      return;

  long int output_idx = ((phase * outputSize.z + k) * outputSize.y + j) * outputSize.x + i;
  long int input_idx = (k * outputSize.y + j) * outputSize.x + i;

  output[output_idx] += input[input_idx] * weight;
}

void
CUDA_splat(const int4 &outputSize,
                   float* input,
                   float* output,
                   int projectionNumber,
                   float **weights)
{

  // Thread Block Dimensions
  int tBlock_x = 16;
  int tBlock_y = 4;
  int tBlock_z = 4;
  int  blocksInX = (outputSize.x - 1) / tBlock_x + 1;
  int  blocksInY = (outputSize.y - 1) / tBlock_y + 1;
  int  blocksInZ = (outputSize.z - 1) / tBlock_z + 1;

  int device;
  cudaGetDevice(&device);

  if(CUDA_VERSION<4000 || GetCudaComputeCapability(device).first<=1)
    {
    dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

    for (int phase=0; phase<outputSize.w; phase++)
      {
      float weight = weights[phase][projectionNumber];
      if(weight!=0)
        {
          splat_kernel <<< dimGrid, dimBlock >>> ( input,
                                                   outputSize,
                                                   output,
                                                   phase,
                                                   weight,
                                                   blocksInX);
        }
      }
    }
  else
    {
    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

    for (int phase=0; phase<outputSize.w; phase++)
      {
      float weight = weights[phase][projectionNumber];
      if(weight!=0)
        {
            splat_kernel_3Dgrid <<< dimGrid, dimBlock >>> ( input,
                                                     outputSize,
                                                     output,
                                                     phase,
                                                     weight);
        }
      }
    }
}
