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
#include "rtkCudaWarpImageFilter.hcu"

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

// CONSTANTS //////////////////////////////////////////////////////////////
__constant__ float c_IndexOutputToPPOutputMatrix[12];
__constant__ float c_IndexOutputToIndexDVFMatrix[12];
__constant__ float c_PPInputToIndexInputMatrix[12];
////////////////////////////////////////////////////////////////////////////

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__global__ void
kernel_3Dgrid(float *             dev_vol_out,
              int3                vol_dim,
              cudaTextureObject_t tex_xdvf,
              cudaTextureObject_t tex_ydvf,
              cudaTextureObject_t tex_zdvf,
              cudaTextureObject_t tex_input_vol)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= vol_dim.x || j >= vol_dim.y || k >= vol_dim.z)
  {
    return;
  }

  // Index row major into the volume
  long int vol_idx = i + (j + k * vol_dim.y) * (vol_dim.x);

  // Matrix multiply to get the index in the DVF texture of the current point in the output volume
  float3 IndexInDVF = matrix_multiply(make_float3(i, j, k), c_IndexOutputToIndexDVFMatrix);

  // Get each component of the displacement vector by
  // interpolation in the dvf
  float3 Displacement;
  Displacement.x = tex3D<float>(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.y = tex3D<float>(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.z = tex3D<float>(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

  // Matrix multiply to get the physical coordinates of the current point in the output volume
  float3 PP = matrix_multiply(make_float3(i, j, k), c_IndexOutputToPPOutputMatrix);

  // Get the index corresponding to the current physical point in output displaced by the displacement vector
  PP += Displacement;

  // Convert it to a continuous index
  float3 IndexInInput = matrix_multiply(PP, c_PPInputToIndexInputMatrix);

  // Interpolate in the input and copy into the output
  dev_vol_out[vol_idx] =
    tex3D<float>(tex_input_vol, IndexInInput.x + 0.5f, IndexInInput.y + 0.5f, IndexInInput.z + 0.5f);
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_warp /////////////////////////////
void
CUDA_warp(int     input_vol_dim[3],
          int     input_dvf_dim[3],
          int     output_vol_dim[3],
          float   IndexOutputToPPOutputMatrix[12],
          float   IndexOutputToIndexDVFMatrix[12],
          float   PPInputToIndexInputMatrix[12],
          float * dev_input_vol,
          float * dev_output_vol,
          float * dev_DVF,
          bool    isLinear)
{
  // Prepare DVF textures
  std::vector<cudaArray *>         DVFComponentArrays;
  std::vector<cudaTextureObject_t> tex_dvf;
  prepareVectorTextureObject(input_dvf_dim, dev_DVF, DVFComponentArrays, 3, tex_dvf, false);

  // Prepare volume texture
  cudaArray *         array_input_vol;
  cudaTextureObject_t tex_input_vol;
  prepareScalarTextureObject(output_vol_dim, dev_input_vol, array_input_vol, tex_input_vol, false, isLinear);

  // Copy matrices into constant memory
  cudaMemcpyToSymbol(
    c_IndexOutputToPPOutputMatrix, IndexOutputToPPOutputMatrix, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(
    c_IndexOutputToIndexDVFMatrix, IndexOutputToIndexDVFMatrix, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(
    c_PPInputToIndexInputMatrix, PPInputToIndexInputMatrix, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR;

  // Thread Block Dimensions
  constexpr int tBlock_x = 16;
  constexpr int tBlock_y = 4;
  constexpr int tBlock_z = 4;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int blocksInX = (output_vol_dim[0] - 1) / tBlock_x + 1;
  unsigned int blocksInY = (output_vol_dim[1] - 1) / tBlock_y + 1;
  unsigned int blocksInZ = (output_vol_dim[2] - 1) / tBlock_z + 1;

  dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

  kernel_3Dgrid<<<dimGrid, dimBlock>>>(dev_output_vol,
                                       make_int3(output_vol_dim[0], output_vol_dim[1], output_vol_dim[2]),
                                       tex_dvf[0],
                                       tex_dvf[1],
                                       tex_dvf[2],
                                       tex_input_vol);
  CUDA_CHECK_ERROR;

  // Cleanup
  for (unsigned int c = 0; c < 3; c++)
  {
    cudaFreeArray(DVFComponentArrays[c]);
    CUDA_CHECK_ERROR;
    cudaDestroyTextureObject(tex_dvf[c]);
    CUDA_CHECK_ERROR;
  }
  cudaFreeArray(array_input_vol);
  CUDA_CHECK_ERROR;
  cudaDestroyTextureObject(tex_input_vol);
  CUDA_CHECK_ERROR;
}
