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
#include "rtkCudaAverageOutOfROIImageFilter.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>
#include <cuda_runtime.h>

// TEXTURES AND CONSTANTS //

__constant__ int4 c_Size;

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_


__global__
void
average_along_dim_4(float *in, float *out, float *roi, unsigned int strideInFloats)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= c_Size.x || j >= c_Size.y || k >= c_Size.z)
      return;

  // Compute the index of the initial voxel
  long int id = (k * c_Size.y + j) * c_Size.x + i;
  long int strided_id = id; // strided_id will run along the 4th dimension

  // Compute the average along last dimension
  float avg = 0;
  for (unsigned int l = 0; l < c_Size.w; l++)
    {
      avg += in[strided_id];
      strided_id += strideInFloats;
    }
  avg /= c_Size.w;

  // Write to the output. If the ROI is 0, replace by the average. If it is 1,
  // do nothing. Since I do not trust CUDA's comparisons between floats,
  // I wrote it without ifs. It has the side effect that the ROI can be non-binary
  // without any problem
  strided_id = id;
  for (unsigned int l = 0; l < c_Size.w; l++)
    {
      out[strided_id] = in[strided_id] * roi[id] + avg * (1-roi[id]);
      strided_id += strideInFloats;
    }

}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

void
CUDA_average_out_of_ROI(int size[4],
                        float* input,
                        float* output,
                        float* roi)
{
  int4 dev_Size = make_int4(size[0], size[1], size[2], size[3]);
  cudaMemcpyToSymbol(c_Size, &dev_Size, sizeof(int4));

  // Compute the stride (in floats, not Bytes) to jump from one voxel
  // to the next one along 4th dimension
  unsigned int strideInFloats = size[0] * size[1] * size[2];

  // Thread Block Dimensions
  dim3 dimBlock = dim3(8, 8, 8);

  int blocksInX = iDivUp(size[0], dimBlock.x);
  int blocksInY = iDivUp(size[1], dimBlock.y);
  int blocksInZ = iDivUp(size[2], dimBlock.z);

  dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);

  average_along_dim_4 <<< dimGrid, dimBlock >>> ( input, output, roi, strideInFloats);
}
