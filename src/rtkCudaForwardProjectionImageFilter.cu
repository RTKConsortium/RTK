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

// TEXTURES AND CONSTANTS //

texture<float, 3, cudaReadModeElementType> tex_vol;

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
__global__
void kernel_forwardProject(float *dev_proj_in, float *dev_proj_out)
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

    // Detect intersection with box
    if ( !intersectBox(ray, &tnear, &tfar, c_boxMin, c_boxMax) || tfar < 0.f )
      {
      dev_proj_out[numThread + proj * c_projSize.x * c_projSize.y] = dev_proj_in[numThread + proj * c_projSize.x * c_projSize.y];
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
      float  sample = 0.0f;
      float  sum    = 0.0f;
      for(t=tnear; t<=tfar; t+=vStep)
        {
        // Read from 3D texture from volume
        sample = tex3D(tex_vol, pos.x, pos.y, pos.z);

        sum += sample;
        pos += step;
        }
      dev_proj_out[numThread + proj * c_projSize.x * c_projSize.y] = dev_proj_in[numThread + proj * c_projSize.x * c_projSize.y] + (sum+(tfar-t+halfVStep)/vStep*sample) * c_tStep;
      }
    }
}

__device__
float notex3D(float *vol, float3 pos, int3 volSize)
{
  int3 floor_pos;
  floor_pos.x = floor(pos.x);
  floor_pos.y = floor(pos.y);
  floor_pos.z = floor(pos.z);

  // Compute the weights
  float weights[8];

  float3 Distance;
  Distance.x = pos.x - floor_pos.x;
  Distance.y = pos.y - floor_pos.y;
  Distance.z = pos.z - floor_pos.z;

  weights[0] = (1 - Distance.x) * (1 - Distance.y) * (1 - Distance.z); //weight000
  weights[1] = (1 - Distance.x) * (1 - Distance.y) * Distance.z;       //weight001
  weights[2] = (1 - Distance.x) * Distance.y       * (1 - Distance.z); //weight010
  weights[3] = (1 - Distance.x) * Distance.y       * Distance.z;       //weight011
  weights[4] = Distance.x       * (1 - Distance.y) * (1 - Distance.z); //weight100
  weights[5] = Distance.x       * (1 - Distance.y) * Distance.z;       //weight101
  weights[6] = Distance.x       * Distance.y       * (1 - Distance.z); //weight110
  weights[7] = Distance.x       * Distance.y       * Distance.z;       //weight111

  // Compute positions of sampling, taking into account the clamping
  float3 pos_low;
  pos_low.x = max(static_cast<float>(floor_pos.x), 0.0f);
  pos_low.y = max(static_cast<float>(floor_pos.y), 0.0f);
  pos_low.z = max(static_cast<float>(floor_pos.z), 0.0f);

  float3 pos_high;
  pos_high.x = min(static_cast<float>(floor_pos.x + 1), static_cast<float>( volSize.x - 1) );
  pos_high.y = min(static_cast<float>(floor_pos.y + 1), static_cast<float>( volSize.y - 1) );
  pos_high.z = min(static_cast<float>(floor_pos.z + 1), static_cast<float>( volSize.z - 1) );

  // Compute indices in the volume
  long int indices[8];

  indices[0] = pos_low.x  + pos_low.y  * volSize.x + pos_low.z  * volSize.x * volSize.y; //index000
  indices[1] = pos_low.x  + pos_low.y  * volSize.x + pos_high.z * volSize.x * volSize.y; //index001
  indices[2] = pos_low.x  + pos_high.y * volSize.x + pos_low.z  * volSize.x * volSize.y; //index010
  indices[3] = pos_low.x  + pos_high.y * volSize.x + pos_high.z * volSize.x * volSize.y; //index011
  indices[4] = pos_high.x + pos_low.y  * volSize.x + pos_low.z  * volSize.x * volSize.y; //index100
  indices[5] = pos_high.x + pos_low.y  * volSize.x + pos_high.z * volSize.x * volSize.y; //index101
  indices[6] = pos_high.x + pos_high.y * volSize.x + pos_low.z  * volSize.x * volSize.y; //index110
  indices[7] = pos_high.x + pos_high.y * volSize.x + pos_high.z * volSize.x * volSize.y; //index111

  float out=0;

  // Perform interpolation
  for (unsigned int i=0; i<8; i++)
    out += vol[indices[i]] * weights[i];

  return out;
}

__global__
void kernel_forwardProject_noTexture(float *dev_proj_in, float *dev_proj_out, float *dev_vol)
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

    // Detect intersection with box
    if ( !intersectBox(ray, &tnear, &tfar, c_boxMin, c_boxMax) || tfar < 0.f )
      {
      dev_proj_out[numThread + proj * c_projSize.x * c_projSize.y] = dev_proj_in[numThread + proj * c_projSize.x * c_projSize.y];
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
      float  sample = 0.0f;
      float  sum    = 0.0f;
      for(t=tnear; t<=tfar; t+=vStep)
        {
        // Read from 3D texture from volume
        sample = notex3D(dev_vol, pos, c_volSize);

        sum += sample;
        pos += step;
        }
      dev_proj_out[numThread + proj * c_projSize.x * c_projSize.y] = dev_proj_in[numThread + proj * c_projSize.x * c_projSize.y] + (sum+(tfar-t+halfVStep)/vStep*sample) * c_tStep;
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
                      bool useCudaTexture)
{
  // Constant memory
//  float3 dev_boxMin = make_float3(box_min[0], box_min[1], box_min[2]);
//  float3 dev_boxMax = make_float3(box_max[0], box_max[1], box_max[2]);
//  float3 dev_spacing = make_float3(spacing[0], spacing[1], spacing[2]);
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

  if (useCudaTexture)
    {
    // Set texture parameters
    tex_vol.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex_vol.addressMode[1] = cudaAddressModeClamp;
    tex_vol.addressMode[2] = cudaAddressModeClamp;
    tex_vol.normalized = false;                     // access with normalized texture coordinates
    tex_vol.filterMode = cudaFilterModeLinear;      // linear interpolation

    // Copy volume data to array, bind the array to the texture
    cudaExtent volExtent = make_cudaExtent(volSize[0], volSize[1], volSize[2]);
    cudaArray *array_vol;
    static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMalloc3DArray((cudaArray**)&array_vol, &channelDesc, volExtent);
    CUDA_CHECK_ERROR;

    // Copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(dev_vol, volSize[0]*sizeof(float), volSize[0], volSize[1]);
    copyParams.dstArray = (cudaArray*)array_vol;
    copyParams.extent   = volExtent;
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);
    CUDA_CHECK_ERROR;

    // Bind 3D array to 3D texture
    cudaBindTextureToArray(tex_vol, (cudaArray*)array_vol, channelDesc);
    CUDA_CHECK_ERROR;

    // Run the kernel
    kernel_forwardProject <<< dimGrid, dimBlock >>> (dev_proj_in, dev_proj_out);

    cudaUnbindTexture (tex_vol);
    cudaFreeArray ((cudaArray*)array_vol);
    CUDA_CHECK_ERROR;
    }
  else
    {
    // Run the kernel without using texture memory
    kernel_forwardProject_noTexture <<< dimGrid, dimBlock >>> (dev_proj_in, dev_proj_out, dev_vol);
    }
}
