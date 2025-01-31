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

/*****************
 *  rtk #includes *
 *****************/
#include "rtkCudaUtilities.hcu"
#include "rtkConfiguration.h"
#include "rtkCudaIntersectBox.hcu"
#include "rtkCudaWarpForwardProjectionImageFilter.hcu"

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

// CONSTANTS
__constant__ int3   c_projSize;
__constant__ float3 c_boxMin;
__constant__ float3 c_boxMax;
__constant__ float3 c_spacing;
__constant__ int3   c_volSize;
__constant__ float  c_tStep;
__constant__ float  c_matrices[SLAB_SIZE * 12]; // Can process stacks of at most SLAB_SIZE projections
__constant__ float  c_sourcePos[SLAB_SIZE * 3]; // Can process stacks of at most SLAB_SIZE projections

__constant__ float c_IndexInputToPPInputMatrix[12];
__constant__ float c_IndexInputToIndexDVFMatrix[12];
__constant__ float c_PPInputToIndexInputMatrix[12];

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// KERNEL kernel_forwardProject
__global__ void
kernel_warped_forwardProject(float *             dev_proj_in,
                             float *             dev_proj_out,
                             cudaTextureObject_t tex_xdvf,
                             cudaTextureObject_t tex_ydvf,
                             cudaTextureObject_t tex_zdvf,
                             cudaTextureObject_t tex_vol)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int numThread = j * c_projSize.x + i;

  if (i >= c_projSize.x || j >= c_projSize.y)
    return;

  // Setting ray origin
  Ray    ray;
  float3 pixelPos;
  float  tnear, tfar;

  for (unsigned int proj = 0; proj < c_projSize.z; proj++)
  {
    // Setting ray origin
    ray.o = make_float3(c_sourcePos[3 * proj], c_sourcePos[3 * proj + 1], c_sourcePos[3 * proj + 2]);

    pixelPos = matrix_multiply(make_float3(i, j, 0), &(c_matrices[12 * proj]));

    ray.d = pixelPos - ray.o;
    ray.d = ray.d / sqrtf(dot(ray.d, ray.d));

    // Detect intersection with box
    if (!intersectBox(ray, &tnear, &tfar, c_boxMin, c_boxMax) || tfar < 0.f)
    {
      dev_proj_out[numThread + proj * c_projSize.x * c_projSize.y] =
        dev_proj_in[numThread + proj * c_projSize.x * c_projSize.y];
    }
    else
    {
      if (tnear < 0.f)
        tnear = 0.f; // clamp to near plane

      // Step length in mm
      float3 dirInMM = c_spacing * ray.d;
      float  vStep = c_tStep / sqrtf(dot(dirInMM, dirInMM));
      float3 step = vStep * ray.d;

      // First position in the box
      float halfVStep = 0.5f * vStep;
      tnear = tnear + halfVStep;
      float3 pos = ray.o + tnear * ray.d;

      float t;
      float sample = 0.0f;
      float sum = 0.0f;

      float3 IndexInDVF, Displacement, PP, IndexInInput;

      for (t = tnear; t <= tfar; t += vStep)
      {
        IndexInDVF = matrix_multiply(pos, c_IndexInputToIndexDVFMatrix);

        // Get each component of the displacement vector by
        // interpolation in the dvf
        Displacement.x = tex3D<float>(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
        Displacement.y = tex3D<float>(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
        Displacement.z = tex3D<float>(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

        // Matrix multiply to get the physical coordinates of the current point in the output volume
        // + the displacement
        PP = matrix_multiply(pos, c_IndexInputToPPInputMatrix) + Displacement;

        // Convert it to a continuous index
        IndexInInput = matrix_multiply(PP, c_PPInputToIndexInputMatrix);

        // Read from 3D texture from volume
        sample = tex3D<float>(tex_vol, IndexInInput.x, IndexInInput.y, IndexInInput.z);

        // Accumulate, and move forward along the ray
        sum += sample;
        pos += step;
      }
      dev_proj_out[numThread + proj * c_projSize.x * c_projSize.y] =
        dev_proj_in[numThread + proj * c_projSize.x * c_projSize.y] +
        (sum + (tfar - t + halfVStep) / vStep * sample) * c_tStep;
    }
  }
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_forward_project() //////////////////////////////////
void
CUDA_warp_forward_project(int     projSize[3],
                          int     volSize[3],
                          int     dvfSize[3],
                          float * matrices,
                          float * dev_proj_in,
                          float * dev_proj_out,
                          float * dev_vol,
                          float   t_step,
                          float * source_positions,
                          float   box_min[3],
                          float   box_max[3],
                          float   spacing[3],
                          float * dev_input_dvf,
                          float   IndexInputToIndexDVFMatrix[12],
                          float   PPInputToIndexInputMatrix[12],
                          float   IndexInputToPPInputMatrix[12])
{
  // constant memory
  cudaMemcpyToSymbol(c_projSize, projSize, sizeof(int3));
  cudaMemcpyToSymbol(c_boxMin, box_min, sizeof(float3));
  cudaMemcpyToSymbol(c_boxMax, box_max, sizeof(float3));
  cudaMemcpyToSymbol(c_spacing, spacing, sizeof(float3));
  cudaMemcpyToSymbol(c_volSize, volSize, sizeof(int3));
  cudaMemcpyToSymbol(c_tStep, &t_step, sizeof(float));

  // Copy the source position matrix into a float3 in constant memory
  cudaMemcpyToSymbol(c_sourcePos, &(source_positions[0]), 3 * sizeof(float) * projSize[2]);

  // Copy the projection matrices into constant memory
  cudaMemcpyToSymbol(c_matrices, &(matrices[0]), 12 * sizeof(float) * projSize[2]);

  // Prepare volume texture
  cudaArray *         array_vol;
  cudaTextureObject_t tex_vol;
  prepareScalarTextureObject(volSize, dev_vol, array_vol, tex_vol, false, true, cudaAddressModeClamp);

  // Prepare DVF textures
  std::vector<cudaArray *>         DVFComponentArrays;
  std::vector<cudaTextureObject_t> tex_dvf;
  prepareVectorTextureObject(dvfSize, dev_input_dvf, DVFComponentArrays, 3, tex_dvf, false);

  // Copy matrices into constant memory
  cudaMemcpyToSymbol(
    c_IndexInputToPPInputMatrix, IndexInputToPPInputMatrix, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(
    c_IndexInputToIndexDVFMatrix, IndexInputToIndexDVFMatrix, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(
    c_PPInputToIndexInputMatrix, PPInputToIndexInputMatrix, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);

  ///////////////
  // RUN
  dim3 dimBlock = dim3(16, 16, 1);
  dim3 dimGrid = dim3(iDivUp(projSize[0], dimBlock.x), iDivUp(projSize[1], dimBlock.y));

  kernel_warped_forwardProject<<<dimGrid, dimBlock>>>(
    dev_proj_in, dev_proj_out, tex_dvf[0], tex_dvf[1], tex_dvf[2], tex_vol);

  // Cleanup
  for (unsigned int c = 0; c < 3; c++)
  {
    cudaFreeArray(DVFComponentArrays[c]);
    CUDA_CHECK_ERROR;
    cudaDestroyTextureObject(tex_dvf[c]);
    CUDA_CHECK_ERROR;
  }
  cudaFreeArray((cudaArray *)array_vol);
  CUDA_CHECK_ERROR;
  cudaDestroyTextureObject(tex_vol);
  CUDA_CHECK_ERROR;
}
