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
#include "rtkCudaRayCastBackProjectionImageFilter.hcu"

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
__constant__ bool c_normalize;

//__constant__ float3 spacingSquare;  // inverse view matrix

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__device__
void splat3D_getWeightsAndIndices(float3 pos, int3 floor_pos, int3 volSize, float *weights, long int *indices)
{
  // Compute the weights
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
  int3 pos_low;
  pos_low.x = max(floor_pos.x, 0);
  pos_low.y = max(floor_pos.y, 0);
  pos_low.z = max(floor_pos.z, 0);

  int3 pos_high;
  pos_high.x = min(floor_pos.x + 1, volSize.x - 1);
  pos_high.y = min(floor_pos.y + 1, volSize.y - 1);
  pos_high.z = min(floor_pos.z + 1, volSize.z - 1);

  // Compute indices in the volume
  indices[0] = pos_low.x  + pos_low.y  * volSize.x + pos_low.z  * volSize.x * volSize.y; //index000
  indices[1] = pos_low.x  + pos_low.y  * volSize.x + pos_high.z * volSize.x * volSize.y; //index001
  indices[2] = pos_low.x  + pos_high.y * volSize.x + pos_low.z  * volSize.x * volSize.y; //index010
  indices[3] = pos_low.x  + pos_high.y * volSize.x + pos_high.z * volSize.x * volSize.y; //index011
  indices[4] = pos_high.x + pos_low.y  * volSize.x + pos_low.z  * volSize.x * volSize.y; //index100
  indices[5] = pos_high.x + pos_low.y  * volSize.x + pos_high.z * volSize.x * volSize.y; //index101
  indices[6] = pos_high.x + pos_high.y * volSize.x + pos_low.z  * volSize.x * volSize.y; //index110
  indices[7] = pos_high.x + pos_high.y * volSize.x + pos_high.z * volSize.x * volSize.y; //index111
}

__device__
void splat3D(float toSplat,
             float *dev_accumulate_values,
             float *dev_accumulate_weights,
             float *weights,
             long int *indices)
{
  // Perform splat
  for (unsigned int i=0; i<8; i++)
    {
    atomicAdd(&dev_accumulate_values[indices[i]], toSplat * weights[i]);
    atomicAdd(&dev_accumulate_weights[indices[i]], weights[i]);
    }
}


// KERNEL normalize
__global__
void kernel_normalize_and_add_to_output(float * dev_vol_in, float * dev_vol_out, float * dev_accumulate_weights, float * dev_accumulate_values)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= c_volSize.x || j >= c_volSize.y || k >= c_volSize.z)
    {
    return;
    }

  // Index row major into the volume
  long int out_idx = i + (j + k*c_volSize.y)*(c_volSize.x);

  float eps = 1e-6;

  // Divide the output volume's voxels by the accumulated splat weights
//   unless the accumulated splat weights are equal to zero
  if (c_normalize)
    {
    if (abs(dev_accumulate_weights[out_idx]) > eps)
      dev_vol_out[out_idx] = dev_vol_in[out_idx] + (dev_accumulate_values[out_idx] / dev_accumulate_weights[out_idx]);
    else
      dev_vol_out[out_idx] = dev_vol_in[out_idx];
    }
  else
    dev_vol_out[out_idx] = dev_vol_in[out_idx] + dev_accumulate_values[out_idx];
}

// KERNEL kernel_ray_cast_back_project
__global__
void kernel_ray_cast_back_project(float *dev_accumulate_values,  float *dev_proj, float * dev_accumulate_weights)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int numThread = j*c_projSize.x + i;
  unsigned int proj = blockIdx.z*blockDim.z + threadIdx.z;

  if (i >= c_projSize.x || j >= c_projSize.y || proj >= c_projSize.z)
    return;

  // Declare variables used in the loop
  Ray ray;
  float3 pixelPos;
  float tnear, tfar;

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
  if ( intersectBox(ray, &tnear, &tfar, c_boxMin, c_boxMax) && !(tfar < 0.f) )
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

    float toSplat;
    long int indices[8];
    float weights[8];
    int3 floor_pos;

    if (tfar - tnear > halfVStep)
      {
      for(t=tnear; t<=tfar; t+=vStep)
        {
        floor_pos.x = floor(pos.x);
        floor_pos.y = floor(pos.y);
        floor_pos.z = floor(pos.z);

        // Compute the weights and the voxel indices, taking into account border conditions (here clamping)
        splat3D_getWeightsAndIndices(pos, floor_pos, c_volSize, weights, indices);

        // Compute the value to be splatted
        toSplat = dev_proj[numThread + proj * c_projSize.x * c_projSize.y] * c_tStep;
        splat3D(toSplat, dev_accumulate_values, dev_accumulate_weights, weights, indices);

        // Move to next position
        pos += step;
        }

      // Last position
      toSplat = dev_proj[numThread + proj * c_projSize.x * c_projSize.y] * c_tStep * (tfar - t + halfVStep) / vStep;
      splat3D(toSplat, dev_accumulate_values, dev_accumulate_weights, weights, indices);
      }
    }
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_ray_cast_backproject() //////////////////////////////////
void
CUDA_ray_cast_back_project( int projSize[3],
                      int volSize[3],
                      float* translatedProjectionIndexTransformMatrices,
                      float* translatedVolumeTransformMatrices,
                      float *dev_vol_in,
                      float *dev_vol_out,
                      float *dev_proj,
                      float t_step,
                      double* source_positions,
                      float radiusCylindricalDetector,
                      float box_min[3],
                      float box_max[3],
                      float spacing[3],
                      bool normalize)
{
  // Constant memory
  cudaMemcpyToSymbol(c_projSize, projSize, sizeof(int3));
  cudaMemcpyToSymbol(c_boxMin, box_min, sizeof(float3));
  cudaMemcpyToSymbol(c_boxMax, box_max, sizeof(float3));
  cudaMemcpyToSymbol(c_spacing, spacing, sizeof(float3));
  cudaMemcpyToSymbol(c_volSize, volSize, sizeof(int3));
  cudaMemcpyToSymbol(c_tStep, &t_step, sizeof(float));
  cudaMemcpyToSymbol(c_radius, &radiusCylindricalDetector, sizeof(float));
  cudaMemcpyToSymbol(c_normalize, &normalize, sizeof(bool));

  // Copy the source position matrix into a float3 in constant memory
  cudaMemcpyToSymbol(c_sourcePos, &(source_positions[0]), 3 * sizeof(float) * projSize[2]);

  // Copy the projection matrices into constant memory
  cudaMemcpyToSymbol(c_translatedProjectionIndexTransformMatrices, &(translatedProjectionIndexTransformMatrices[0]), 12 * sizeof(float) * projSize[2]);
  cudaMemcpyToSymbol(c_translatedVolumeTransformMatrices, &(translatedVolumeTransformMatrices[0]), 12 * sizeof(float) * projSize[2]);

  // Create an image to store the splatted values
  // We cannot use the output image, because it may not be zero, in which case
  // normalization by the splat weights would affect not only the backprojection
  // of the current projection, but also the initial value of the output
  float *dev_accumulate_values;
  cudaMalloc( (void**)&dev_accumulate_values, sizeof(float) * volSize[0] * volSize[1] * volSize[2]);
  cudaMemset((void *)dev_accumulate_values, 0, sizeof(float) * volSize[0] * volSize[1] * volSize[2]);

  // Create an image to store the splat weights (in order to normalize)
  float *dev_accumulate_weights;
  cudaMalloc( (void**)&dev_accumulate_weights, sizeof(float) * volSize[0] * volSize[1] * volSize[2]);
  cudaMemset((void *)dev_accumulate_weights, 0, sizeof(float) * volSize[0] * volSize[1] * volSize[2]);

  // Calling kernels
  dim3 dimBlock  = dim3(8, 8, 4);
  dim3 dimGrid = dim3(iDivUp(projSize[0], dimBlock.x), iDivUp(projSize[1], dimBlock.y), iDivUp(projSize[2], dimBlock.z));

  kernel_ray_cast_back_project <<< dimGrid, dimBlock >>> (dev_accumulate_values, dev_proj, dev_accumulate_weights);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR;

  dim3 dimBlockVol = dim3(16, 4, 4);
  dim3 dimGridVol = dim3(iDivUp(volSize[0], dimBlockVol.x), iDivUp(volSize[1], dimBlockVol.y), iDivUp(volSize[2], dimBlockVol.z));

  kernel_normalize_and_add_to_output <<< dimGridVol, dimBlockVol >>> ( dev_vol_in, dev_vol_out, dev_accumulate_weights, dev_accumulate_values);

  CUDA_CHECK_ERROR;

  // Cleanup
  cudaFree (dev_accumulate_weights);
  cudaFree (dev_accumulate_values);
  CUDA_CHECK_ERROR;
}
