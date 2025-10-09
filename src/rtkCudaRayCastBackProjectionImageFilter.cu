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
#include "rtkCudaRayCastBackProjectionImageFilter.hcu"

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

// TEXTURES AND CONSTANTS //

__constant__ int3   c_projSize;
__constant__ float3 c_boxMin;
__constant__ float3 c_boxMax;
__constant__ float3 c_spacing;
__constant__ int3   c_volSize;
__constant__ float  c_tStep;
__constant__ float  c_radius;
__constant__ float
  c_translatedProjectionIndexTransformMatrices[SLAB_SIZE * 12]; // Can process stacks of at most SLAB_SIZE projections
__constant__ float
  c_translatedVolumeTransformMatrices[SLAB_SIZE * 12]; // Can process stacks of at most SLAB_SIZE projections
__constant__ float c_sourcePos[SLAB_SIZE * 3];         // Can process stacks of at most SLAB_SIZE projections

__global__ void
kernel_ray_cast_back_project(float * dev_vol_out, float * dev_proj)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int numThread = j * c_projSize.x + i;
  unsigned int proj = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= c_projSize.x || j >= c_projSize.y || proj >= c_projSize.z)
    return;

  // Declare variables used in the loop
  Ray    ray;
  float3 pixelPos;
  float  tnear, tfar;

  // Setting ray origin
  ray.o = make_float3(c_sourcePos[3 * proj], c_sourcePos[3 * proj + 1], c_sourcePos[3 * proj + 2]);

  if (c_radius == 0)
  {
    pixelPos = matrix_multiply(make_float3(i, j, 0), &(c_translatedProjectionIndexTransformMatrices[12 * proj]));
  }
  else
  {
    float3 posProj;
    posProj = matrix_multiply(make_float3(i, j, 0), &(c_translatedProjectionIndexTransformMatrices[12 * proj]));
    double a = posProj.x / c_radius;
    posProj.x = sin(a) * c_radius;
    posProj.z += (1. - cos(a)) * c_radius;
    pixelPos = matrix_multiply(posProj, &(c_translatedVolumeTransformMatrices[12 * proj]));
  }

  ray.d = pixelPos - ray.o;
  ray.d = ray.d / sqrtf(dot(ray.d, ray.d));

  // Detect intersection with box
  if (intersectBox(ray, &tnear, &tfar, c_boxMin, c_boxMax) && tfar > 0.f && tfar != tnear)
  {
    if (tnear < 0.f)
      tnear = 0.f; // clamp to near plane

    // Step length in mm
    float3 dirInMM = c_spacing * ray.d;
    float  dirLengthInMM = sqrtf(dot(dirInMM, dirInMM));
    float  vStep = min(c_tStep / dirLengthInMM, tfar - tnear);
    float  mmStep = vStep * dirLengthInMM;

    // Skip rays intersecting less half a step
    float    toSplat;
    long int indices[8];
    float    weights[8];
    int3     floor_pos;

    // First position in the box
    float halfVStep = 0.5f * vStep;
    tnear = tnear + halfVStep;

    float3 pos = ray.o + tnear * ray.d;
    float3 step = vStep * ray.d;
    float  t;
    for (t = tnear; t <= tfar; t += vStep)
    {
      floor_pos.x = floorf(pos.x);
      floor_pos.y = floorf(pos.y);
      floor_pos.z = floorf(pos.z);

      // Compute the weights
      float3 Distance;
      Distance.x = pos.x - floor_pos.x;
      Distance.y = pos.y - floor_pos.y;
      Distance.z = pos.z - floor_pos.z;

      weights[0] = (1 - Distance.x) * (1 - Distance.y) * (1 - Distance.z); // weight000
      weights[1] = (1 - Distance.x) * (1 - Distance.y) * Distance.z;       // weight001
      weights[2] = (1 - Distance.x) * Distance.y * (1 - Distance.z);       // weight010
      weights[3] = (1 - Distance.x) * Distance.y * Distance.z;             // weight011
      weights[4] = Distance.x * (1 - Distance.y) * (1 - Distance.z);       // weight100
      weights[5] = Distance.x * (1 - Distance.y) * Distance.z;             // weight101
      weights[6] = Distance.x * Distance.y * (1 - Distance.z);             // weight110
      weights[7] = Distance.x * Distance.y * Distance.z;                   // weight111

      // Compute positions of sampling, taking into account the clamping
      int3 pos_low;
      pos_low.x = max(floor_pos.x, 0);
      pos_low.y = max(floor_pos.y, 0);
      pos_low.z = max(floor_pos.z, 0);

      int3 pos_high;
      pos_high.x = min(floor_pos.x + 1, c_volSize.x - 1);
      pos_high.y = min(floor_pos.y + 1, c_volSize.y - 1);
      pos_high.z = min(floor_pos.z + 1, c_volSize.z - 1);

      // Compute indices in the volume
      indices[0] = pos_low.x + pos_low.y * c_volSize.x + pos_low.z * c_volSize.x * c_volSize.y;    // index000
      indices[1] = pos_low.x + pos_low.y * c_volSize.x + pos_high.z * c_volSize.x * c_volSize.y;   // index001
      indices[2] = pos_low.x + pos_high.y * c_volSize.x + pos_low.z * c_volSize.x * c_volSize.y;   // index010
      indices[3] = pos_low.x + pos_high.y * c_volSize.x + pos_high.z * c_volSize.x * c_volSize.y;  // index011
      indices[4] = pos_high.x + pos_low.y * c_volSize.x + pos_low.z * c_volSize.x * c_volSize.y;   // index100
      indices[5] = pos_high.x + pos_low.y * c_volSize.x + pos_high.z * c_volSize.x * c_volSize.y;  // index101
      indices[6] = pos_high.x + pos_high.y * c_volSize.x + pos_low.z * c_volSize.x * c_volSize.y;  // index110
      indices[7] = pos_high.x + pos_high.y * c_volSize.x + pos_high.z * c_volSize.x * c_volSize.y; // index111

      // Compute the value to be splatted
      toSplat = dev_proj[numThread + proj * c_projSize.x * c_projSize.y] * mmStep;
      atomicAdd(&dev_vol_out[indices[0]], toSplat * weights[0]);
      atomicAdd(&dev_vol_out[indices[1]], toSplat * weights[1]);
      atomicAdd(&dev_vol_out[indices[2]], toSplat * weights[2]);
      atomicAdd(&dev_vol_out[indices[3]], toSplat * weights[3]);
      atomicAdd(&dev_vol_out[indices[4]], toSplat * weights[4]);
      atomicAdd(&dev_vol_out[indices[5]], toSplat * weights[5]);
      atomicAdd(&dev_vol_out[indices[6]], toSplat * weights[6]);
      atomicAdd(&dev_vol_out[indices[7]], toSplat * weights[7]);

      // Move to next position
      pos += step;
    }

    // Last position
    toSplat = dev_proj[numThread + proj * c_projSize.x * c_projSize.y] * (tfar - t + halfVStep) * dirLengthInMM;
    atomicAdd(&dev_vol_out[indices[0]], toSplat * weights[0]);
    atomicAdd(&dev_vol_out[indices[1]], toSplat * weights[1]);
    atomicAdd(&dev_vol_out[indices[2]], toSplat * weights[2]);
    atomicAdd(&dev_vol_out[indices[3]], toSplat * weights[3]);
    atomicAdd(&dev_vol_out[indices[4]], toSplat * weights[4]);
    atomicAdd(&dev_vol_out[indices[5]], toSplat * weights[5]);
    atomicAdd(&dev_vol_out[indices[6]], toSplat * weights[6]);
    atomicAdd(&dev_vol_out[indices[7]], toSplat * weights[7]);
  }
}

void
CUDA_ray_cast_back_project(int      projSize[3],
                           int      volSize[3],
                           float *  translatedProjectionIndexTransformMatrices,
                           float *  translatedVolumeTransformMatrices,
                           float *  dev_vol_in,
                           float *  dev_vol_out,
                           float *  dev_proj,
                           float    t_step,
                           double * source_positions,
                           float    radiusCylindricalDetector,
                           float    box_min[3],
                           float    box_max[3],
                           float    spacing[3])
{
  // Constant memory
  cudaMemcpyToSymbol(c_projSize, projSize, sizeof(int3));
  cudaMemcpyToSymbol(c_boxMin, box_min, sizeof(float3));
  cudaMemcpyToSymbol(c_boxMax, box_max, sizeof(float3));
  cudaMemcpyToSymbol(c_spacing, spacing, sizeof(float3));
  cudaMemcpyToSymbol(c_volSize, volSize, sizeof(int3));
  cudaMemcpyToSymbol(c_tStep, &t_step, sizeof(float));
  cudaMemcpyToSymbol(c_radius, &radiusCylindricalDetector, sizeof(float));

  // Copy the source position matrix into a float3 in constant memory
  cudaMemcpyToSymbol(c_sourcePos, &(source_positions[0]), 3 * sizeof(float) * projSize[2]);

  // Copy the projection matrices into constant memory
  cudaMemcpyToSymbol(c_translatedProjectionIndexTransformMatrices,
                     &(translatedProjectionIndexTransformMatrices[0]),
                     12 * sizeof(float) * projSize[2]);
  cudaMemcpyToSymbol(
    c_translatedVolumeTransformMatrices, &(translatedVolumeTransformMatrices[0]), 12 * sizeof(float) * projSize[2]);

  // If not in place, input must be copied to output
  if (dev_vol_in != dev_vol_out)
  {
    size_t volMemSize = sizeof(float) * volSize[0] * volSize[1] * volSize[2];
    cudaMemcpy(dev_vol_out, dev_vol_in, volMemSize, cudaMemcpyDeviceToDevice);
    CUDA_CHECK_ERROR;
  }

  // Calling kernels
  dim3 dimBlock = dim3(8, 8, 4);
  dim3 dimGrid =
    dim3(iDivUp(projSize[0], dimBlock.x), iDivUp(projSize[1], dimBlock.y), iDivUp(projSize[2], dimBlock.z));

  kernel_ray_cast_back_project<<<dimGrid, dimBlock>>>(dev_vol_out, dev_proj);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR;
}
