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
#include "rtkConfiguration.h"
#include "rtkCudaUtilities.hcu"
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

texture<float, 1, cudaReadModeElementType> tex_matrix;

__constant__ float3 c_sourcePos;
__constant__ int2 c_projSize;
__constant__ float3 c_boxMin;
__constant__ float3 c_boxMax;
__constant__ float3 c_spacing;
__constant__ float c_tStep;
__constant__ int3 c_volSize;
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
void kernel_normalize_and_add_to_output(float * dev_vol_out, float * dev_accumulate_weights, float * dev_accumulate_values, bool normalize)
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
  // unless the accumulated splat weights are equal to zero
  if (normalize)
    {
    if (abs(dev_accumulate_weights[out_idx]) > eps)
      dev_vol_out[out_idx] += (dev_accumulate_values[out_idx] / dev_accumulate_weights[out_idx]);
    }
  else
    dev_vol_out[out_idx] += dev_accumulate_values[out_idx];
}

// KERNEL kernel_ray_cast_back_project
__global__
void kernel_ray_cast_back_project(float *dev_accumulate_values,  float *dev_proj, float * dev_accumulate_weights)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int numThread = j*c_projSize.x + i;

  if (i >= c_projSize.x || j >= c_projSize.y)
    return;

  // Setting ray origin
  Ray ray;
  ray.o = c_sourcePos;

  float3 pixelPos;

  pixelPos.x = tex1Dfetch(tex_matrix, 3)  + tex1Dfetch(tex_matrix, 0)*i +
               tex1Dfetch(tex_matrix, 1)*j;
  pixelPos.y = tex1Dfetch(tex_matrix, 7)  + tex1Dfetch(tex_matrix, 4)*i +
               tex1Dfetch(tex_matrix, 5)*j;
  pixelPos.z = tex1Dfetch(tex_matrix, 11) + tex1Dfetch(tex_matrix, 8)*i +
               tex1Dfetch(tex_matrix, 9)*j;

  ray.d = pixelPos - ray.o;
  ray.d = ray.d / sqrtf(dot(ray.d,ray.d));

  // Detect intersection with box
  float tnear, tfar;
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
        toSplat = dev_proj[numThread] * c_tStep;
        splat3D(toSplat, dev_accumulate_values, dev_accumulate_weights, weights, indices);

        // Move to next position
        pos += step;
        }

      // Last position
      toSplat = dev_proj[numThread] * c_tStep * (tfar - t + halfVStep) / vStep;
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
CUDA_ray_cast_back_project( int projections_size[2],
                      int vol_size[3],
                      float matrix[12],
                      float *dev_vol_out,
                      float *dev_proj,
                      float t_step,
                      double source_position[3],
                      float box_min[3],
                      float box_max[3],
                      float spacing[3],
                      bool normalize)
{
  // Copy matrix and bind data to the texture
  float *dev_matrix;
  cudaMalloc( (void**)&dev_matrix, 12*sizeof(float) );
  cudaMemcpy (dev_matrix, matrix, 12*sizeof(float), cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR;
  cudaBindTexture (0, tex_matrix, dev_matrix, 12*sizeof(float) );
  CUDA_CHECK_ERROR;

  // Create an image to store the splatted values
  // We cannot use the output image, because it may not be zero, in which case
  // normalization by the splat weights would affect not only the backprojection
  // of the current projection, but also the initial value of the output
  float *dev_accumulate_values;
  cudaMalloc( (void**)&dev_accumulate_values, sizeof(float) * vol_size[0] * vol_size[1] * vol_size[2]);
  cudaMemset((void *)dev_accumulate_values, 0, sizeof(float) * vol_size[0] * vol_size[1] * vol_size[2]);

  // Create an image to store the splat weights (in order to normalize)
  float *dev_accumulate_weights;
  cudaMalloc( (void**)&dev_accumulate_weights, sizeof(float) * vol_size[0] * vol_size[1] * vol_size[2]);
  cudaMemset((void *)dev_accumulate_weights, 0, sizeof(float) * vol_size[0] * vol_size[1] * vol_size[2]);

  // constant memory
  float3 dev_sourcePos = make_float3(source_position[0], source_position[1], source_position[2]);
  float3 dev_boxMin = make_float3(box_min[0], box_min[1], box_min[2]);
  int2 dev_projSize = make_int2(projections_size[0], projections_size[1]);
  float3 dev_boxMax = make_float3(box_max[0], box_max[1], box_max[2]);
  float3 dev_spacing = make_float3(spacing[0], spacing[1], spacing[2]);
  int3 dev_volSize = make_int3(vol_size[0], vol_size[1], vol_size[2]);
  cudaMemcpyToSymbol(c_sourcePos, &dev_sourcePos, sizeof(float3));
  cudaMemcpyToSymbol(c_projSize, &dev_projSize, sizeof(int2));
  cudaMemcpyToSymbol(c_boxMin, &dev_boxMin, sizeof(float3));
  cudaMemcpyToSymbol(c_boxMax, &dev_boxMax, sizeof(float3));
  cudaMemcpyToSymbol(c_spacing, &dev_spacing, sizeof(float3));
  cudaMemcpyToSymbol(c_tStep, &t_step, sizeof(float));
  cudaMemcpyToSymbol(c_volSize, &dev_volSize, sizeof(int3));

  // Calling kernels
  dim3 dimBlock  = dim3(16, 16, 1);
  dim3 dimGrid = dim3(iDivUp(projections_size[0], dimBlock.x), iDivUp(projections_size[1], dimBlock.y));

  kernel_ray_cast_back_project <<< dimGrid, dimBlock >>> (dev_accumulate_values, dev_proj, dev_accumulate_weights);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR;

  dim3 dimBlockVol = dim3(16, 4, 4);
  dim3 dimGridVol = dim3(iDivUp(vol_size[0], dimBlockVol.x), iDivUp(vol_size[1], dimBlockVol.y), iDivUp(vol_size[2], dimBlockVol.z));

  kernel_normalize_and_add_to_output <<< dimGridVol, dimBlockVol >>> ( dev_vol_out, dev_accumulate_weights, dev_accumulate_values, normalize);

  CUDA_CHECK_ERROR;

  // Unbind the volume and matrix textures
  cudaUnbindTexture (tex_matrix);
  CUDA_CHECK_ERROR;

  // Cleanup
  cudaFree (dev_accumulate_weights);
  cudaFree (dev_accumulate_values);
  CUDA_CHECK_ERROR;
  cudaFree (dev_matrix);
  CUDA_CHECK_ERROR;
}
