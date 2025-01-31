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
#include "rtkCudaBackProjectionImageFilter.hcu"

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
__constant__ float c_volIndexToProjPP[SLAB_SIZE * 12];
__constant__ float c_projPPToProjIndex[9];
__constant__ int3  c_projSize;
__constant__ int3  c_volSize;

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

template <unsigned int VVectorLength, bool VIsCylindrical>
__global__ void
kernel_backProject(float * dev_vol_in, float * dev_vol_out, float radius, cudaTextureObject_t * dev_tex_proj)
{
  itk::SizeValueType i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  itk::SizeValueType j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  itk::SizeValueType k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= c_volSize.x || j >= c_volSize.y || k >= c_volSize.z)
  {
    return;
  }

  // Index row major into the volume
  itk::SizeValueType vol_idx = i + (j + k * c_volSize.y) * (c_volSize.x);

  float3 ip, pp;
  float  voxel_data[VVectorLength];
  for (unsigned int c = 0; c < VVectorLength; c++)
    voxel_data[c] = 0.0f;

  for (unsigned int proj = 0; proj < c_projSize.z; proj++)
  {
    if (VIsCylindrical)
    {
      // matrix multiply
      pp = matrix_multiply(make_float3(i, j, k), &(c_volIndexToProjPP[12 * proj]));

      // Change coordinate systems
      pp.z = 1 / pp.z;
      pp.x = pp.x * pp.z;
      pp.y = pp.y * pp.z;

      // Apply correction for cylindrical detector
      const float u = pp.x;
      pp.x = radius * atan2(u, radius);
      pp.y = pp.y * radius / sqrt(radius * radius + u * u);

      // Get projection index
      ip.x = c_projPPToProjIndex[0] * pp.x + c_projPPToProjIndex[1] * pp.y + c_projPPToProjIndex[2];
      ip.y = c_projPPToProjIndex[3] * pp.x + c_projPPToProjIndex[4] * pp.y + c_projPPToProjIndex[5];
    }
    else
    {
      // matrix multiply
      ip = matrix_multiply(make_float3(i, j, k), &(c_matrices[12 * proj]));

      // Change coordinate systems
      ip.z = 1 / ip.z;
      ip.x = ip.x * ip.z;
      ip.y = ip.y * ip.z;
    }

    // Get texture point, clip left to GPU, and accumulate in voxel_data
    for (unsigned int c = 0; c < VVectorLength; c++)
      voxel_data[c] += tex2DLayered<float>(dev_tex_proj[c], ip.x, ip.y, proj);
  }

  // Place it into the volume
  for (unsigned int c = 0; c < VVectorLength; c++)
    dev_vol_out[vol_idx * VVectorLength + c] = dev_vol_in[vol_idx * VVectorLength + c] + voxel_data[c];
}


//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_back_project /////////////////////////////
void
CUDA_back_project(int          projSize[3],
                  int          volSize[3],
                  float *      matrices,
                  float *      volIndexToProjPPs,
                  float *      projPPToProjIndex,
                  float *      dev_vol_in,
                  float *      dev_vol_out,
                  float *      dev_proj,
                  double       radiusCylindricalDetector,
                  unsigned int vectorLength)
{
  // Copy the size of inputs into constant memory
  cudaMemcpyToSymbol(c_projSize, projSize, sizeof(int3));
  cudaMemcpyToSymbol(c_volSize, volSize, sizeof(int3));

  // Copy the projection matrices into constant memory
  cudaMemcpyToSymbol(c_matrices, &(matrices[0]), 12 * sizeof(float) * projSize[2]);
  cudaMemcpyToSymbol(c_volIndexToProjPP, &(volIndexToProjPPs[0]), 12 * sizeof(float) * projSize[2]);
  cudaMemcpyToSymbol(c_projPPToProjIndex, &(projPPToProjIndex[0]), 9 * sizeof(float));

  // Thread Block Dimensions
  constexpr int tBlock_x = 16;
  constexpr int tBlock_y = 4;
  constexpr int tBlock_z = 4;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int blocksInX = (volSize[0] - 1) / tBlock_x + 1;
  unsigned int blocksInY = (volSize[1] - 1) / tBlock_y + 1;
  unsigned int blocksInZ = (volSize[2] - 1) / tBlock_z + 1;

  // Run kernels. Note: Projection data is passed via texture memory,
  // transform matrix is passed via constant memory

  // Compute block and grid sizes
  dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
  CUDA_CHECK_ERROR;

  // Prepare texture objects (needs an array of cudaTextureObjects on the host as "tex_proj" argument)
  std::vector<cudaArray *>         projComponentArrays;
  std::vector<cudaTextureObject_t> tex_proj;
  prepareVectorTextureObject(projSize, dev_proj, projComponentArrays, vectorLength, tex_proj, true);

  // Copy them to a device pointer, since it will have to be de-referenced in the kernels
  cudaTextureObject_t * dev_tex_proj;
  cudaMalloc(&dev_tex_proj, vectorLength * sizeof(cudaTextureObject_t));
  cudaMemcpy(dev_tex_proj, tex_proj.data(), vectorLength * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

  // Run the kernel. Since "vectorLength" is passed as a function argument, not as a template argument,
  // the compiler can't assume it's constant, and a dirty trick has to be used.
  // I did not manage to make CUDA_forward_project templated over vectorLength,
  // which would be the best solution
  // Since the list of arguments is long, and is the same in all cases (only the template parameters
  // change), it is passed by a macro
  switch (vectorLength)
  {
    case 1:
      if (radiusCylindricalDetector == 0)
        kernel_backProject<1, false>
          <<<dimGrid, dimBlock>>>(dev_vol_in, dev_vol_out, (float)radiusCylindricalDetector, dev_tex_proj);
      else
        kernel_backProject<1, true>
          <<<dimGrid, dimBlock>>>(dev_vol_in, dev_vol_out, (float)radiusCylindricalDetector, dev_tex_proj);
      break;

    case 2:
      if (radiusCylindricalDetector == 0)
        kernel_backProject<2, false>
          <<<dimGrid, dimBlock>>>(dev_vol_in, dev_vol_out, (float)radiusCylindricalDetector, dev_tex_proj);
      else
        kernel_backProject<2, true>
          <<<dimGrid, dimBlock>>>(dev_vol_in, dev_vol_out, (float)radiusCylindricalDetector, dev_tex_proj);
      break;

    case 3:
      if (radiusCylindricalDetector == 0)
        kernel_backProject<3, false>
          <<<dimGrid, dimBlock>>>(dev_vol_in, dev_vol_out, (float)radiusCylindricalDetector, dev_tex_proj);
      else
        kernel_backProject<3, true>
          <<<dimGrid, dimBlock>>>(dev_vol_in, dev_vol_out, (float)radiusCylindricalDetector, dev_tex_proj);
      break;

    case 4:
      if (radiusCylindricalDetector == 0)
        kernel_backProject<4, false>
          <<<dimGrid, dimBlock>>>(dev_vol_in, dev_vol_out, (float)radiusCylindricalDetector, dev_tex_proj);
      else
        kernel_backProject<4, true>
          <<<dimGrid, dimBlock>>>(dev_vol_in, dev_vol_out, (float)radiusCylindricalDetector, dev_tex_proj);
      break;

    case 9:
      if (radiusCylindricalDetector == 0)
        kernel_backProject<9, false>
          <<<dimGrid, dimBlock>>>(dev_vol_in, dev_vol_out, (float)radiusCylindricalDetector, dev_tex_proj);
      else
        kernel_backProject<9, true>
          <<<dimGrid, dimBlock>>>(dev_vol_in, dev_vol_out, (float)radiusCylindricalDetector, dev_tex_proj);
      break;

    default:
    {
      itkGenericExceptionMacro("Vector length " << vectorLength << " is not supported.");
    }
  }
  CUDA_CHECK_ERROR;

  // Cleanup
  for (unsigned int c = 0; c < vectorLength; c++)
  {
    cudaFreeArray(projComponentArrays[c]);
    CUDA_CHECK_ERROR;
    cudaDestroyTextureObject(tex_proj[c]);
    CUDA_CHECK_ERROR;
  }
  cudaFree(dev_tex_proj);
  CUDA_CHECK_ERROR;
}
