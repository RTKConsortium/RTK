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

/*****************
*  rtk #includes *
*****************/
#include "rtkCudaUtilities.hcu"
#include "rtkConfiguration.h"
#include "rtkCudaIntersectBox.hcu"
#include "rtkCudaForwardProjectionImageFilter.hcu"

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
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CONSTANTS //
__constant__ int3 c_projSize;
__constant__ float3 c_boxMin;
__constant__ float3 c_boxMax;
__constant__ float3 c_spacing;
__constant__ int3 c_volSize;
__constant__ float c_tStep;
__constant__ float c_radius;
__constant__ float c_translatedProjectionIndexTransformMatrices[SLAB_SIZE * 12]; //Can process stacks of at most SLAB_SIZE projections
__constant__ float c_translatedVolumeTransformMatrices[SLAB_SIZE * 12]; //Can process stacks of at most SLAB_SIZE projections
__constant__ float c_sourcePos[SLAB_SIZE * 3]; //Can process stacks of at most SLAB_SIZE projections


//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// KERNEL kernel_forwardProject
template<unsigned int vectorLength>
__global__
void kernel_forwardProject(float *dev_proj_in, float *dev_proj_out, float* dev_vol, cudaTextureObject_t* dev_tex_vol)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int numThread = j*c_projSize.x + i;

  if (i >= c_projSize.x || j >= c_projSize.y)
    return;

  // Declare variables used in the loop
  Ray ray;
  float3 pixelPos;
  float tnear, tfar;

  for (unsigned int proj = 0; proj<c_projSize.z; proj++)
    {
    // Setting ray origin
    ray.o = make_float3(c_sourcePos[3 * proj], c_sourcePos[3 * proj + 1], c_sourcePos[3 * proj + 2]);

    if (c_radius == 0)
      {
      pixelPos = matrix_multiply(make_float3(i,j,0), &(c_translatedProjectionIndexTransformMatrices[12*proj]));
      }
    else
      {
      float3 posProj;
      posProj = matrix_multiply(make_float3(i,j,0), &(c_translatedProjectionIndexTransformMatrices[12*proj]));
      double a = posProj.x / c_radius;
      posProj.x = sin(a) * c_radius;
      posProj.z += (1. - cos(a)) * c_radius;
      pixelPos = matrix_multiply(posProj, &(c_translatedVolumeTransformMatrices[12*proj]));
      }

    ray.d = pixelPos - ray.o;
    ray.d = ray.d / sqrtf(dot(ray.d,ray.d));

    int projOffset = numThread + proj * c_projSize.x * c_projSize.y;

    // Detect intersection with box
    if ( !intersectBox(ray, &tnear, &tfar, c_boxMin, c_boxMax) || tfar < 0.f )
      {
      for (unsigned int c = 0; c < vectorLength; c++)
        dev_proj_out[projOffset * vectorLength + c] = dev_proj_in[projOffset * vectorLength + c];
      }
    else
      {
      if (tnear < 0.f)
        tnear = 0.f; // clamp to near plane

      // Step length in mm
      float3 dirInMM = c_spacing * ray.d;
      float vStep = c_tStep / sqrtf(dot(dirInMM, dirInMM));
      float3 step = vStep * ray.d;

      // First position in the box
      float3 pos;
      float halfVStep = 0.5f*vStep;
      tnear = tnear + halfVStep;
      pos = ray.o + tnear*ray.d;

      float  t;
      float sample[vectorLength];
      float sum[vectorLength];
      for (unsigned int c=0; c<vectorLength; c++)
        {
        sample[c] = 0.0f;
        sum[c]    = 0.0f;
        }

      for(t=tnear; t<=tfar; t+=vStep)
        {
        // Read from 3D texture from volume(s)
        for (unsigned int c=0; c<vectorLength; c++)
          sample[c] = tex3D<float>(dev_tex_vol[c], pos.x, pos.y, pos.z);

        // Accumulate
        for (unsigned int c=0; c<vectorLength; c++)
          sum[c] += sample[c];

        // Step forward
        pos += step;
        }

      // Update the output projection pixels
      for (unsigned int c = 0; c < vectorLength; c++)
        dev_proj_out[projOffset * vectorLength + c] = dev_proj_in[projOffset * vectorLength + c] + (sum[c] + (tfar-t+halfVStep) / vStep * sample[c]) * c_tStep;
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
CUDA_forward_project(int projSize[3],
                      int volSize[3],
                      float* translatedProjectionIndexTransformMatrices,
                      float* translatedVolumeTransformMatrices,
                      float *dev_proj_in,
                      float *dev_proj_out,
                      float *dev_vol,
                      float t_step,
                      float* source_positions,
                      float radiusCylindricalDetector,
                      float box_min[3],
                      float box_max[3],
                      float spacing[3],
                      unsigned int vectorLength)
{
  // Constant memory
  cudaMemcpyToSymbol(c_projSize, projSize, sizeof(int3));
  cudaMemcpyToSymbol(c_boxMin, box_min, sizeof(float3));
  cudaMemcpyToSymbol(c_boxMax, box_max, sizeof(float3));
  cudaMemcpyToSymbol(c_spacing, spacing, sizeof(float3));
  cudaMemcpyToSymbol(c_volSize, volSize, sizeof(int3));
  cudaMemcpyToSymbol(c_tStep, &t_step, sizeof(float));
  cudaMemcpyToSymbol(c_radius, &radiusCylindricalDetector, sizeof(float));

  dim3 dimBlock  = dim3(16, 16, 1);
  dim3 dimGrid = dim3(iDivUp(projSize[0], dimBlock.x), iDivUp(projSize[1], dimBlock.x));

  // Copy the source position matrix into a float3 in constant memory
  cudaMemcpyToSymbol(c_sourcePos, &(source_positions[0]), 3 * sizeof(float) * projSize[2]);

  // Copy the projection matrices into constant memory
  cudaMemcpyToSymbol(c_translatedProjectionIndexTransformMatrices, &(translatedProjectionIndexTransformMatrices[0]), 12 * sizeof(float) * projSize[2]);
  cudaMemcpyToSymbol(c_translatedVolumeTransformMatrices, &(translatedVolumeTransformMatrices[0]), 12 * sizeof(float) * projSize[2]);

  // Create an array of textures
  cudaTextureObject_t* tex_vol = new cudaTextureObject_t[vectorLength];
  cudaArray** volComponentArrays = new cudaArray* [vectorLength];

  // Prepare texture objects (needs an array of cudaTextureObjects on the host as "tex_vol" argument)
  prepareTextureObject(volSize, dev_vol, volComponentArrays, vectorLength, tex_vol, false);

  // Copy them to a device pointer, since it will have to be de-referenced in the kernels
  cudaTextureObject_t* dev_tex_vol;
  cudaMalloc(&dev_tex_vol, vectorLength * sizeof(cudaTextureObject_t));
  cudaMemcpy(dev_tex_vol, tex_vol, vectorLength * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

  // Run the kernel. Since "vectorLength" is passed as a function argument, not as a template argument,
  // the compiler can't assume it's constant, and a dirty trick has to be used.
  // I did not manage to make CUDA_forward_project templated over vectorLength,
  // which would be the best solution
  switch(vectorLength)
    {
    case 1:
    kernel_forwardProject<1> <<< dimGrid, dimBlock >>> (dev_proj_in, dev_proj_out, dev_vol, dev_tex_vol);
    break;

    case 3:
    kernel_forwardProject<3> <<< dimGrid, dimBlock >>> (dev_proj_in, dev_proj_out, dev_vol, dev_tex_vol);
    break;
    }
  CUDA_CHECK_ERROR;

  // Cleanup
  for (unsigned int c=0; c<vectorLength; c++)
    {
    cudaFreeArray ((cudaArray*) volComponentArrays[c]);
    cudaDestroyTextureObject(tex_vol[c]);
    }
  cudaFree(dev_tex_vol);
  delete[] volComponentArrays;
  CUDA_CHECK_ERROR;

  delete[] tex_vol;
}
