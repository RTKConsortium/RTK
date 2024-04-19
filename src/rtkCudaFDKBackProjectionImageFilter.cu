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

/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/*****************
 *  rtk #includes *
 *****************/
#include "rtkCudaUtilities.hcu"
#include "rtkConfiguration.h"
#include "rtkCudaFDKBackProjectionImageFilter.hcu"

/*****************
 *  C   #includes *
 *****************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/*****************
 * CUDA #includes *
 *****************/
#include <cuda.h>

// Constant memory
__constant__ float c_matrices[SLAB_SIZE * 12]; // Can process stacks of at most SLAB_SIZE projections
__constant__ int3 c_projSize;
__constant__ int3 c_vol_size;

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__global__ void
kernel_fdk_3Dgrid(float * dev_vol_in, float * dev_vol_out, cudaTextureObject_t tex_proj)
{
  itk::SizeValueType i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  itk::SizeValueType j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  itk::SizeValueType k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= c_vol_size.x || j >= c_vol_size.y || k >= c_vol_size.z)
  {
    return;
  }

  // Index row major into the volume
  itk::SizeValueType vol_idx = i + (j + k * c_vol_size.y) * (c_vol_size.x);

  float3 ip;
  float  voxel_data = 0;

  for (unsigned int proj = 0; proj < c_projSize.z; proj++)
  {
    // matrix multiply
    ip = matrix_multiply(make_float3(i, j, k), &(c_matrices[12 * proj]));

    // Change coordinate systems
    ip.z = 1 / ip.z;
    ip.x = ip.x * ip.z;
    ip.y = ip.y * ip.z;

    // Get texture point, clip left to GPU, and accumulate in voxel_data
    voxel_data += tex2DLayered<float>(tex_proj, ip.x, ip.y, proj) * ip.z * ip.z;
  }

  // Place it into the volume
  dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel_data;
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_back_project /////////////////////////////
void
CUDA_reconstruct_conebeam(int     proj_size[3],
                          int     vol_size[3],
                          float * matrices,
                          float * dev_vol_in,
                          float * dev_vol_out,
                          float * dev_proj)
{
  // Copy the size of inputs into constant memory
  cudaMemcpyToSymbol(c_projSize, proj_size, sizeof(int3));
  cudaMemcpyToSymbol(c_vol_size, vol_size, sizeof(int3));

  // Copy the projection matrices into constant memory
  cudaMemcpyToSymbol(c_matrices, &(matrices[0]), 12 * sizeof(float) * proj_size[2]);

  // Thread Block Dimensions
  constexpr int tBlock_x = 16;
  constexpr int tBlock_y = 4;
  constexpr int tBlock_z = 4;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int blocksInX = (vol_size[0] - 1) / tBlock_x + 1;
  unsigned int blocksInY = (vol_size[1] - 1) / tBlock_y + 1;
  unsigned int blocksInZ = (vol_size[2] - 1) / tBlock_z + 1;

  // Run kernels. Note: Projection data is passed via texture memory,
  // transform matrix is passed via constant memory

  // Compute block and grid sizes
  dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

  cudaArray *         array_proj;
  cudaTextureObject_t tex_proj;
  prepareScalarTextureObject(proj_size, dev_proj, array_proj, tex_proj, true);
  kernel_fdk_3Dgrid<<<dimGrid, dimBlock>>>(dev_vol_in, dev_vol_out, tex_proj);

  // Cleanup
  cudaFreeArray((cudaArray *)array_proj);
  CUDA_CHECK_ERROR;
  cudaDestroyTextureObject(tex_proj);
  CUDA_CHECK_ERROR;
}
