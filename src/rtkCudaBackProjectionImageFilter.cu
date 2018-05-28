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

/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/*****************
*  rtk #includes *
*****************/
#include "rtkCudaUtilities.hcu"
#include "rtkConfiguration.h"
#include "rtkCudaBackProjectionImageFilter.hcu"

/*****************
*  C   #includes *
*****************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*****************
* CUDA #includes *
*****************/
#include <cuda.h>

// T E X T U R E S ////////////////////////////////////////////////////////
static cudaTextureObject_t tex_proj[3];

// Constant memory
__constant__ float c_matrices[SLAB_SIZE * 12]; //Can process stacks of at most SLAB_SIZE projections
__constant__ float c_volIndexToProjPP[SLAB_SIZE * 12];
__constant__ float c_projPPToProjIndex[9];
__constant__ int3 c_projSize;
__constant__ int3 c_volSize;

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

template<unsigned int vectorLength, bool isCylindrical>
__global__
void kernel_backProject(float *dev_vol_in, float * dev_vol_out, double radius, cudaTextureObject_t tex_proj0, cudaTextureObject_t tex_proj1, cudaTextureObject_t tex_proj2)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= c_volSize.x || j >= c_volSize.y || k >= c_volSize.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*c_volSize.y)*(c_volSize.x);

  float3 ip, pp;
  float voxel_data[vectorLength];
  for (unsigned int c=0; c<vectorLength; c++)
    voxel_data[c] = 0.0f;

  for (unsigned int proj = 0; proj<c_projSize.z; proj++)
    {
    if (isCylindrical)
      {
      // matrix multiply
      pp = matrix_multiply(make_float3(i,j,k), &(c_volIndexToProjPP[12*proj]));

      // Change coordinate systems
      pp.z = 1 / pp.z;
      pp.x = pp.x * pp.z;
      pp.y = pp.y * pp.z;

      // Apply correction for cylindrical detector
      double u = pp.x;
      pp.x = radius * atan(u / radius);
      pp.y = pp.y * radius / sqrt(radius * radius + u * u);

      // Get projection index
      ip.x = c_projPPToProjIndex[0 * 3 + 0] * pp.x + c_projPPToProjIndex[0 * 3 + 1] * pp.y + c_projPPToProjIndex[0 * 3 + 2];
      ip.y = c_projPPToProjIndex[1 * 3 + 0] * pp.x + c_projPPToProjIndex[1 * 3 + 1] * pp.y + c_projPPToProjIndex[1 * 3 + 2];
      }
    else
      {
      // matrix multiply
      ip = matrix_multiply(make_float3(i,j,k), &(c_matrices[12*proj]));

      // Change coordinate systems
      ip.z = 1 / ip.z;
      ip.x = ip.x * ip.z;
      ip.y = ip.y * ip.z;
      }

    // Get texture point, clip left to GPU, and accumulate in voxel_data
    switch(vectorLength)
      {
      case 1:
        voxel_data[0] += tex2DLayered<float>(tex_proj0, ip.x, ip.y, proj);
        break;
      case 3:
        voxel_data[0] += tex2DLayered<float>(tex_proj0, ip.x, ip.y, proj);
        voxel_data[1] += tex2DLayered<float>(tex_proj1, ip.x, ip.y, proj);
        voxel_data[2] += tex2DLayered<float>(tex_proj2, ip.x, ip.y, proj);
        break;
      }
    }

  // Place it into the volume
  for (unsigned int c=0; c<vectorLength; c++)
    dev_vol_out[vol_idx * vectorLength + c] = dev_vol_in[vol_idx * vectorLength + c] + voxel_data[c];
}


//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_back_project /////////////////////////////
void
CUDA_back_project(int projSize[3],
                  int volSize[3],
                  float *matrices,
                  float *volIndexToProjPPs,
                  float *projPPToProjIndex,
                  float *dev_vol_in,
                  float *dev_vol_out,
                  float *dev_proj,
                  double radiusCylindricalDetector,
                  unsigned int vectorLength)
{
  // 2D layered texture requires CudaComputeCapability >= 2.0
  int device;
  cudaGetDevice(&device);
  if(GetCudaComputeCapability(device).first<=1)
    itkGenericExceptionMacro(<< "RTK no longer supports GPUs with CudaComputeCapability < 2.0")

  // Copy the size of inputs into constant memory
  cudaMemcpyToSymbol(c_projSize, projSize, sizeof(int3));
  cudaMemcpyToSymbol(c_volSize, volSize, sizeof(int3));

  // Copy the projection matrices into constant memory
  cudaMemcpyToSymbol(c_matrices,          &(matrices[0]),          12 * sizeof(float) * projSize[2]);
  cudaMemcpyToSymbol(c_volIndexToProjPP,  &(volIndexToProjPPs[0]), 12 * sizeof(float) * projSize[2]);
  cudaMemcpyToSymbol(c_projPPToProjIndex, &(projPPToProjIndex[0]), 9 * sizeof(float));

  // Thread Block Dimensions
  const int tBlock_x = 16;
  const int tBlock_y = 4;
  const int tBlock_z = 4;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int  blocksInX = (volSize[0]-1)/tBlock_x + 1;
  unsigned int  blocksInY = (volSize[1]-1)/tBlock_y + 1;
  unsigned int  blocksInZ = (volSize[2]-1)/tBlock_z + 1;

  // Run kernels. Note: Projection data is passed via texture memory,
  // transform matrix is passed via constant memory

  // Compute block and grid sizes
  dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
  CUDA_CHECK_ERROR;

  cudaArray** projComponentArrays = new cudaArray* [vectorLength];

  // Prepare texture objects
  prepareTextureObject(projSize, dev_proj, projComponentArrays, vectorLength, tex_proj, true);

  // Run the kernel. Since "vectorLength" is passed as a function argument, not as a template argument,
  // the compiler can't assume it's constant, and a dirty trick has to be used.
  // I did not manage to make CUDA_forward_project templated over vectorLength,
  // which would be the best solution
  switch(vectorLength)
    {
    case 1:
      if (radiusCylindricalDetector == 0)
        kernel_backProject<1, false> <<< dimGrid, dimBlock >>> (dev_vol_in, dev_vol_out, radiusCylindricalDetector, tex_proj[0], tex_proj[1], tex_proj[2]);
      else
        kernel_backProject<1, true> <<< dimGrid, dimBlock >>> (dev_vol_in, dev_vol_out, radiusCylindricalDetector, tex_proj[0], tex_proj[1], tex_proj[2]);
    break;

    case 3:
      if (radiusCylindricalDetector == 0)
        kernel_backProject<3, false> <<< dimGrid, dimBlock >>> (dev_vol_in, dev_vol_out, radiusCylindricalDetector, tex_proj[0], tex_proj[1], tex_proj[2]);
      else
        kernel_backProject<3, true> <<< dimGrid, dimBlock >>> (dev_vol_in, dev_vol_out, radiusCylindricalDetector, tex_proj[0], tex_proj[1], tex_proj[2]);
    break;
    }
  CUDA_CHECK_ERROR;

  // Cleanup
  for (unsigned int c=0; c<vectorLength; c++)
    {
    cudaFreeArray ((cudaArray*) projComponentArrays[c]);
    cudaDestroyTextureObject(tex_proj[c]);
    }
  delete[] projComponentArrays;
  CUDA_CHECK_ERROR;
}
