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

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
  return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
  return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

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

struct Ray {
        float3 o;  // origin
        float3 d;  // direction
};

inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// Intersection function of a ray with a box, followed "slabs" method
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
__device__
int intersectBox2(Ray r, float *tnear, float *tfar)
{
    // Compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.f / r.d.x, 1.f / r.d.y, 1.f / r.d.z);
    float3 T1;
    T1 = invR * (c_boxMin - r.o);
    float3 T2;
    T2 = invR * (c_boxMax - r.o);

    // Re-order intersections to find smallest and largest on each axis
    float3 tmin;
    tmin = fminf(T2, T1);
    float3 tmax;
    tmax = fmaxf(T2, T1);

    // Find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// KERNEL normalize
__global__
void kernel_normalize_and_add_to_output(float * dev_vol_out, float * dev_accumulate_weights, float * dev_accumulate_values)
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
  if (abs(dev_accumulate_weights[out_idx]) > eps)
    dev_vol_out[out_idx] += (dev_accumulate_values[out_idx] / dev_accumulate_weights[out_idx]);
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
  if ( intersectBox2(ray, &tnear, &tfar) && !(tfar < 0.f) )
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

    bool isInVolume_out_idx000;
    bool isInVolume_out_idx001;
    bool isInVolume_out_idx010;
    bool isInVolume_out_idx011;
    bool isInVolume_out_idx100;
    bool isInVolume_out_idx101;
    bool isInVolume_out_idx110;
    bool isInVolume_out_idx111;

    long int out_idx000;
    long int out_idx001;
    long int out_idx010;
    long int out_idx011;
    long int out_idx100;
    long int out_idx101;
    long int out_idx110;
    long int out_idx111;

    float weight000;
    float weight001;
    float weight010;
    float weight011;
    float weight100;
    float weight101;
    float weight110;
    float weight111;

    for(t=tnear; t<=tfar; t+=vStep)
      {
      // Compute the splat weights
      int3 BaseIndexInOutput;
      BaseIndexInOutput.x = floor(pos.x);
      BaseIndexInOutput.y = floor(pos.y);
      BaseIndexInOutput.z = floor(pos.z);

      float3 Distance;
      Distance.x = pos.x - BaseIndexInOutput.x;
      Distance.y = pos.y - BaseIndexInOutput.y;
      Distance.z = pos.z - BaseIndexInOutput.z;

      weight000 = (1 - Distance.x) * (1 - Distance.y) * (1 - Distance.z);
      weight001 = (1 - Distance.x) * (1 - Distance.y) * Distance.z;
      weight010 = (1 - Distance.x) * Distance.y       * (1 - Distance.z);
      weight011 = (1 - Distance.x) * Distance.y       * Distance.z;
      weight100 = Distance.x       * (1 - Distance.y) * (1 - Distance.z);
      weight101 = Distance.x       * (1 - Distance.y) * Distance.z;
      weight110 = Distance.x       * Distance.y       * (1 - Distance.z);
      weight111 = Distance.x       * Distance.y       * Distance.z;

      // Compute indices in the volume
      out_idx000 = (BaseIndexInOutput.x + 0) + (BaseIndexInOutput.y + 0) * c_volSize.x + (BaseIndexInOutput.z + 0) * c_volSize.x * c_volSize.y;
      out_idx001 = (BaseIndexInOutput.x + 0) + (BaseIndexInOutput.y + 0) * c_volSize.x + (BaseIndexInOutput.z + 1) * c_volSize.x * c_volSize.y;
      out_idx010 = (BaseIndexInOutput.x + 0) + (BaseIndexInOutput.y + 1) * c_volSize.x + (BaseIndexInOutput.z + 0) * c_volSize.x * c_volSize.y;
      out_idx011 = (BaseIndexInOutput.x + 0) + (BaseIndexInOutput.y + 1) * c_volSize.x + (BaseIndexInOutput.z + 1) * c_volSize.x * c_volSize.y;
      out_idx100 = (BaseIndexInOutput.x + 1) + (BaseIndexInOutput.y + 0) * c_volSize.x + (BaseIndexInOutput.z + 0) * c_volSize.x * c_volSize.y;
      out_idx101 = (BaseIndexInOutput.x + 1) + (BaseIndexInOutput.y + 0) * c_volSize.x + (BaseIndexInOutput.z + 1) * c_volSize.x * c_volSize.y;
      out_idx110 = (BaseIndexInOutput.x + 1) + (BaseIndexInOutput.y + 1) * c_volSize.x + (BaseIndexInOutput.z + 0) * c_volSize.x * c_volSize.y;
      out_idx111 = (BaseIndexInOutput.x + 1) + (BaseIndexInOutput.y + 1) * c_volSize.x + (BaseIndexInOutput.z + 1) * c_volSize.x * c_volSize.y;

      // Determine whether they are indeed in the volume
      isInVolume_out_idx000 = (BaseIndexInOutput.x + 0 >= 0) && (BaseIndexInOutput.x + 0 < c_volSize.x)
                                && (BaseIndexInOutput.y + 0 >= 0) && (BaseIndexInOutput.y + 0 < c_volSize.y)
                                && (BaseIndexInOutput.z + 0 >= 0) && (BaseIndexInOutput.z + 0 < c_volSize.z);

      isInVolume_out_idx001 = (BaseIndexInOutput.x + 0 >= 0) && (BaseIndexInOutput.x + 0 < c_volSize.x)
                                && (BaseIndexInOutput.y + 0 >= 0) && (BaseIndexInOutput.y + 0 < c_volSize.y)
                                && (BaseIndexInOutput.z + 1 >= 0) && (BaseIndexInOutput.z + 1 < c_volSize.z);

      isInVolume_out_idx010 = (BaseIndexInOutput.x + 0 >= 0) && (BaseIndexInOutput.x + 0 < c_volSize.x)
                                && (BaseIndexInOutput.y + 1 >= 0) && (BaseIndexInOutput.y + 1 < c_volSize.y)
                                && (BaseIndexInOutput.z + 0 >= 0) && (BaseIndexInOutput.z + 0 < c_volSize.z);

      isInVolume_out_idx011 = (BaseIndexInOutput.x + 0 >= 0) && (BaseIndexInOutput.x + 0 < c_volSize.x)
                                && (BaseIndexInOutput.y + 1 >= 0) && (BaseIndexInOutput.y + 1 < c_volSize.y)
                                && (BaseIndexInOutput.z + 1 >= 0) && (BaseIndexInOutput.z + 1 < c_volSize.z);

      isInVolume_out_idx100 = (BaseIndexInOutput.x + 1 >= 0) && (BaseIndexInOutput.x + 1 < c_volSize.x)
                                && (BaseIndexInOutput.y + 0 >= 0) && (BaseIndexInOutput.y + 0 < c_volSize.y)
                                && (BaseIndexInOutput.z + 0 >= 0) && (BaseIndexInOutput.z + 0 < c_volSize.z);

      isInVolume_out_idx101 = (BaseIndexInOutput.x + 1 >= 0) && (BaseIndexInOutput.x + 1 < c_volSize.x)
                                && (BaseIndexInOutput.y + 0 >= 0) && (BaseIndexInOutput.y + 0 < c_volSize.y)
                                && (BaseIndexInOutput.z + 1 >= 0) && (BaseIndexInOutput.z + 1 < c_volSize.z);

      isInVolume_out_idx110 = (BaseIndexInOutput.x + 1 >= 0) && (BaseIndexInOutput.x + 1 < c_volSize.x)
                                && (BaseIndexInOutput.y + 1 >= 0) && (BaseIndexInOutput.y + 1 < c_volSize.y)
                                && (BaseIndexInOutput.z + 0 >= 0) && (BaseIndexInOutput.z + 0 < c_volSize.z);

      isInVolume_out_idx111 = (BaseIndexInOutput.x + 1 >= 0) && (BaseIndexInOutput.x + 1 < c_volSize.x)
                                && (BaseIndexInOutput.y + 1 >= 0) && (BaseIndexInOutput.y + 1 < c_volSize.y)
                                && (BaseIndexInOutput.z + 1 >= 0) && (BaseIndexInOutput.z + 1 < c_volSize.z);

      // Perform splat if voxel is indeed in output volume
      float toBeWeighed = dev_proj[numThread] * c_tStep;

      if (isInVolume_out_idx000)
        {
          atomicAdd(&dev_accumulate_values[out_idx000], toBeWeighed * weight000);
          atomicAdd(&dev_accumulate_weights[out_idx000], weight000);
        }
      if (isInVolume_out_idx001)
        {
          atomicAdd(&dev_accumulate_values[out_idx001], toBeWeighed * weight001);
          atomicAdd(&dev_accumulate_weights[out_idx001], weight001);
        }
      if (isInVolume_out_idx010)
        {
          atomicAdd(&dev_accumulate_values[out_idx010], toBeWeighed * weight010);
          atomicAdd(&dev_accumulate_weights[out_idx010], weight010);
        }
      if (isInVolume_out_idx011)
        {
          atomicAdd(&dev_accumulate_values[out_idx011], toBeWeighed * weight011);
          atomicAdd(&dev_accumulate_weights[out_idx011], weight011);
        }
      if (isInVolume_out_idx100)
        {
          atomicAdd(&dev_accumulate_values[out_idx100], toBeWeighed * weight100);
          atomicAdd(&dev_accumulate_weights[out_idx100], weight100);
        }
      if (isInVolume_out_idx101)
        {
          atomicAdd(&dev_accumulate_values[out_idx101], toBeWeighed * weight101);
          atomicAdd(&dev_accumulate_weights[out_idx101], weight101);
        }
      if (isInVolume_out_idx110)
        {
          atomicAdd(&dev_accumulate_values[out_idx110], toBeWeighed * weight110);
          atomicAdd(&dev_accumulate_weights[out_idx110], weight110);
        }
      if (isInVolume_out_idx111)
        {
          atomicAdd(&dev_accumulate_values[out_idx111], toBeWeighed * weight111);
          atomicAdd(&dev_accumulate_weights[out_idx111], weight111);
        }

      // Move to next position
      pos += step;
      }

    // Last position
    float toBeWeighed = dev_proj[numThread] * (tfar-t+halfVStep)/vStep * c_tStep;

    if (isInVolume_out_idx000)
      {
        atomicAdd(&dev_accumulate_values[out_idx000], toBeWeighed * weight000);
        atomicAdd(&dev_accumulate_weights[out_idx000], weight000);
      }
    if (isInVolume_out_idx001)
      {
        atomicAdd(&dev_accumulate_values[out_idx001], toBeWeighed * weight001);
        atomicAdd(&dev_accumulate_weights[out_idx001], weight001);
      }
    if (isInVolume_out_idx010)
      {
        atomicAdd(&dev_accumulate_values[out_idx010], toBeWeighed * weight010);
        atomicAdd(&dev_accumulate_weights[out_idx010], weight010);
      }
    if (isInVolume_out_idx011)
      {
        atomicAdd(&dev_accumulate_values[out_idx011], toBeWeighed * weight011);
        atomicAdd(&dev_accumulate_weights[out_idx011], weight011);
      }
    if (isInVolume_out_idx100)
      {
        atomicAdd(&dev_accumulate_values[out_idx100], toBeWeighed * weight100);
        atomicAdd(&dev_accumulate_weights[out_idx100], weight100);
      }
    if (isInVolume_out_idx101)
      {
        atomicAdd(&dev_accumulate_values[out_idx101], toBeWeighed * weight101);
        atomicAdd(&dev_accumulate_weights[out_idx101], weight101);
      }
    if (isInVolume_out_idx110)
      {
        atomicAdd(&dev_accumulate_values[out_idx110], toBeWeighed * weight110);
        atomicAdd(&dev_accumulate_weights[out_idx110], weight110);
      }
    if (isInVolume_out_idx111)
      {
        atomicAdd(&dev_accumulate_values[out_idx111], toBeWeighed * weight111);
        atomicAdd(&dev_accumulate_weights[out_idx111], weight111);
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
                      float spacing[3])
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

  dim3 dimBlockVol = dim3(16, 4, 4);
  dim3 dimGridVol = dim3(iDivUp(vol_size[0], dimBlockVol.x), iDivUp(vol_size[1], dimBlockVol.y), iDivUp(vol_size[2], dimBlockVol.z));

  kernel_normalize_and_add_to_output <<< dimGridVol, dimBlockVol >>> ( dev_vol_out, dev_accumulate_weights, dev_accumulate_values);

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
