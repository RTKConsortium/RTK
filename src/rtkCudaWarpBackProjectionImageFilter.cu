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
#include "rtkCudaWarpBackProjectionImageFilter.hcu"

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
#include <cuda_runtime.h>

// CONSTANTS //////////////////////////////////////////////////////////////
__constant__ float c_matrices[SLAB_SIZE * 12]; // Can process stacks of at most SLAB_SIZE projections
__constant__ float c_volIndexToProjPP[SLAB_SIZE * 12];
__constant__ float c_projPPToProjIndex[9];
__constant__ int3  c_projSize;
__constant__ int3  c_volSize;
__constant__ float c_IndexInputToIndexDVFMatrix[12];
__constant__ float c_PPInputToIndexInputMatrix[12];
__constant__ float c_IndexInputToPPInputMatrix[12];
////////////////////////////////////////////////////////////////////////////

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__global__ void
kernel_warp_back_project_3Dgrid(float *             dev_vol_in,
                                float *             dev_vol_out,
                                cudaTextureObject_t tex_xdvf,
                                cudaTextureObject_t tex_ydvf,
                                cudaTextureObject_t tex_zdvf,
                                cudaTextureObject_t tex_proj)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= c_volSize.x || j >= c_volSize.y || k >= c_volSize.z)
  {
    return;
  }

  // Index row major into the volume
  long int vol_idx = i + (j + k * c_volSize.y) * (c_volSize.x);

  float3 IndexInDVF, Displacement, PP, IndexInInput, ip;
  float  voxel_data = 0;

  for (unsigned int proj = 0; proj < c_projSize.z; proj++)
  {
    // Compute the index in the DVF
    IndexInDVF = matrix_multiply(make_float3(i, j, k), c_IndexInputToIndexDVFMatrix);

    // Get each component of the displacement vector by interpolation in the DVF
    Displacement.x = tex3D<float>(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
    Displacement.y = tex3D<float>(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
    Displacement.z = tex3D<float>(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

    // Compute the physical point in input + the displacement vector
    PP = matrix_multiply(make_float3(i, j, k), c_IndexInputToPPInputMatrix) + Displacement;

    // Convert it to a continuous index
    IndexInInput = matrix_multiply(PP, c_PPInputToIndexInputMatrix);

    // Project the voxel onto the detector to find out which value to add to it
    ip = matrix_multiply(IndexInInput, &(c_matrices[12 * proj]));

    // Change coordinate systems
    ip.z = 1 / ip.z;
    ip.x = ip.x * ip.z;
    ip.y = ip.y * ip.z;

    // Get texture point, clip left to GPU
    voxel_data += tex2DLayered<float>(tex_proj, ip.x, ip.y, proj);
  }

  // Place it into the volume
  dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel_data;
}

__global__ void
kernel_warp_back_project_3Dgrid_cylindrical_detector(float *             dev_vol_in,
                                                     float *             dev_vol_out,
                                                     float               radius,
                                                     cudaTextureObject_t tex_xdvf,
                                                     cudaTextureObject_t tex_ydvf,
                                                     cudaTextureObject_t tex_zdvf,
                                                     cudaTextureObject_t tex_proj)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= c_volSize.x || j >= c_volSize.y || k >= c_volSize.z)
  {
    return;
  }

  // Index row major into the volume
  long int vol_idx = i + (j + k * c_volSize.y) * (c_volSize.x);

  float3 IndexInDVF, Displacement, PP, IndexInInput, ip, pp;
  float  voxel_data = 0;

  for (unsigned int proj = 0; proj < c_projSize.z; proj++)
  {
    // Compute the index in the DVF
    IndexInDVF = matrix_multiply(make_float3(i, j, k), c_IndexInputToIndexDVFMatrix);

    // Get each component of the displacement vector by interpolation in the DVF
    Displacement.x = tex3D<float>(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
    Displacement.y = tex3D<float>(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
    Displacement.z = tex3D<float>(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

    // Compute the physical point in input + the displacement vector
    PP = matrix_multiply(make_float3(i, j, k), c_IndexInputToPPInputMatrix) + Displacement;

    // Convert it to a continuous index
    IndexInInput = matrix_multiply(PP, c_PPInputToIndexInputMatrix);

    // Project the voxel onto the detector to find out which value to add to it
    pp = matrix_multiply(IndexInInput, &(c_volIndexToProjPP[12 * proj]));

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

    // Get texture point, clip left to GPU
    voxel_data += tex2DLayered<float>(tex_proj, ip.x, ip.y, proj);
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
CUDA_warp_back_project(int     projSize[3],
                       int     volSize[3],
                       int     dvf_size[3],
                       float * matrices,
                       float * volIndexToProjPPs,
                       float * projPPToProjIndex,
                       float * dev_vol_in,
                       float * dev_vol_out,
                       float * dev_proj,
                       float * dev_input_dvf,
                       float   IndexInputToIndexDVFMatrix[12],
                       float   PPInputToIndexInputMatrix[12],
                       float   IndexInputToPPInputMatrix[12],
                       double  radiusCylindricalDetector)
{
  // Copy the size of inputs into constant memory
  cudaMemcpyToSymbol(c_projSize, projSize, sizeof(int3));
  cudaMemcpyToSymbol(c_volSize, volSize, sizeof(int3));

  // Copy the projection matrices into constant memory
  cudaMemcpyToSymbol(c_matrices, &(matrices[0]), 12 * sizeof(float) * projSize[2]);
  cudaMemcpyToSymbol(c_volIndexToProjPP, &(volIndexToProjPPs[0]), 12 * sizeof(float) * projSize[2]);
  cudaMemcpyToSymbol(c_projPPToProjIndex, &(projPPToProjIndex[0]), 9 * sizeof(float));

  // Prepare proj texture
  cudaArray *         array_proj;
  cudaTextureObject_t tex_proj;
  prepareScalarTextureObject(projSize, dev_proj, array_proj, tex_proj, true);

  // Prepare DVF textures
  std::vector<cudaArray *>         DVFComponentArrays;
  std::vector<cudaTextureObject_t> tex_dvf;
  prepareVectorTextureObject(dvf_size, dev_input_dvf, DVFComponentArrays, 3, tex_dvf, false);

  // Copy matrices into constant memory
  cudaMemcpyToSymbol(
    c_IndexInputToIndexDVFMatrix, IndexInputToIndexDVFMatrix, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(
    c_PPInputToIndexInputMatrix, PPInputToIndexInputMatrix, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(
    c_IndexInputToPPInputMatrix, IndexInputToPPInputMatrix, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);

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

  // Note: cbi->img is passed via texture memory
  // Matrices are passed via constant memory
  //-------------------------------------
  if (radiusCylindricalDetector == 0)
    kernel_warp_back_project_3Dgrid<<<dimGrid, dimBlock>>>(
      dev_vol_in, dev_vol_out, tex_dvf[0], tex_dvf[1], tex_dvf[2], tex_proj);
  else
    kernel_warp_back_project_3Dgrid_cylindrical_detector<<<dimGrid, dimBlock>>>(
      dev_vol_in, dev_vol_out, (float)radiusCylindricalDetector, tex_dvf[0], tex_dvf[1], tex_dvf[2], tex_proj);
  CUDA_CHECK_ERROR;

  // Cleanup
  for (unsigned int c = 0; c < 3; c++)
  {
    cudaFreeArray(DVFComponentArrays[c]);
    CUDA_CHECK_ERROR;
    cudaDestroyTextureObject(tex_dvf[c]);
    CUDA_CHECK_ERROR;
  }
  cudaFreeArray((cudaArray *)array_proj);
  CUDA_CHECK_ERROR;
  cudaDestroyTextureObject(tex_proj);
  CUDA_CHECK_ERROR;
}
