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

// T E X T U R E S ////////////////////////////////////////////////////////
texture<float, 3, cudaReadModeElementType> tex_vol;
texture<float, 1, cudaReadModeElementType> tex_matrix;
texture<float, 1, cudaReadModeElementType> tex_mu;
///////////////////////////////////////////////////////////////////////////

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
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // Compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.f / r.d.x, 1.f / r.d.y, 1.f / r.d.z);
    float3 T1;
    T1 = invR * (boxmin - r.o);
    float3 T2;
    T2 = invR * (boxmax - r.o);

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

// KERNEL kernel_forwardProject
__global__
void kernel_forwardProject(float *dev_proj,
                           float3 src_pos,
                           int2 proj_dim,
                           float tStep,
                           const float3 boxMin,
                           const float3 boxMax,
                           const float3 spacing)
{

  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (i >= proj_dim.x || j >= proj_dim.y)
    return;

  // Setting ray origin
  Ray ray;
  ray.o = src_pos;

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
  if (!intersectBox(ray, boxMin, boxMax, &tnear, &tfar))
    return;
  if (tnear < 0.f)
    tnear = 0.f; // clamp to near plane

  // Step length in mm
  float3 dirInMM = spacing * ray.d;
  float vStep = tStep / sqrtf(dot(dirInMM, dirInMM));
  float3 step = vStep * ray.d;

  // First position in the box
  float3 pos;
  float halfVStep = 0.5f*vStep;
  tnear = tnear + halfVStep;
  pos = ray.o + tnear*ray.d;

  // Condition to exit the loop
  if(tnear>tfar)
    {
    dev_proj[j*proj_dim.x + i] = 0.0f;
    return;
    }

  float  t;
  float  sample = 0.0f;
  float  sum    = 0.0f;
  for(t=tnear; t<=tfar; t+=vStep)
    {

    // Read from 3D texture from volume, and make a trilinear interpolation
    float xtex = pos.x - 0.5; int itex = floor(xtex); float dxtex = xtex - itex;
    float ytex = pos.y - 0.5; int jtex = floor(ytex); float dytex = ytex - jtex;
    float ztex = pos.z - 0.5; int ktex = floor(ztex); float dztex = ztex - ktex;
    sample =
        (1-dxtex) * (1-dytex) * (1-dztex) * tex1Dfetch(tex_mu, (int)tex3D(tex_vol, itex  , jtex  , ktex))
      + dxtex     * (1-dytex) * (1-dztex) * tex1Dfetch(tex_mu, (int)tex3D(tex_vol, itex+1, jtex  , ktex))
      + (1-dxtex) * dytex     * (1-dztex) * tex1Dfetch(tex_mu, (int)tex3D(tex_vol, itex  , jtex+1, ktex))
      + dxtex     * dytex     * (1-dztex) * tex1Dfetch(tex_mu, (int)tex3D(tex_vol, itex+1, jtex+1, ktex))
      + (1-dxtex) * (1-dytex) * dztex     * tex1Dfetch(tex_mu, (int)tex3D(tex_vol, itex  , jtex  , ktex+1))
      + dxtex     * (1-dytex) * dztex     * tex1Dfetch(tex_mu, (int)tex3D(tex_vol, itex+1, jtex  , ktex+1))
      + (1-dxtex) * dytex     * dztex     * tex1Dfetch(tex_mu, (int)tex3D(tex_vol, itex  , jtex+1, ktex+1))
      + dxtex     * dytex     * dztex     * tex1Dfetch(tex_mu, (int)tex3D(tex_vol, itex+1, jtex+1, ktex+1));

    //sample = tex3D(tex_vol, pos.x, pos.y, pos.z);

    sum += sample;
    pos += step;
    }
  dev_proj[j*proj_dim.x + i] = (sum+(tfar-t+halfVStep)*sample) * tStep;
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_forward_project() //////////////////////////////////
void
CUDA_forward_project( int projections_size[2],
                      int vol_size[3],
                      float matrix[12],
                      float mu[256],
                      float *dev_proj,
                      float *dev_vol,
                      float t_step,
                      double source_position[3],
                      float box_min[3],
                      float box_max[3],
                      float spacing[3])
{
  // Set texture parameters
  tex_vol.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
  tex_vol.addressMode[1] = cudaAddressModeClamp;
  tex_vol.normalized = false;                      // access with normalized texture coordinates
  tex_vol.filterMode = cudaFilterModePoint;       // linear interpolation

  // Copy volume data to array, bind the array to the texture
  cudaExtent volExtent =  make_cudaExtent(vol_size[0], vol_size[1], vol_size[2]);
  cudaArray *array_vol;
  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaMalloc3DArray((cudaArray**)&array_vol, &channelDesc, volExtent);

  // Copy data to 3D array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr   = make_cudaPitchedPtr(dev_vol, vol_size[0]*sizeof(float), vol_size[0], vol_size[1]);
  copyParams.dstArray = (cudaArray*)array_vol;
  copyParams.extent   = volExtent;
  copyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&copyParams);

  // Bind 3D array to 3D texture
  cudaBindTextureToArray(tex_vol, (cudaArray*)array_vol, channelDesc);

  static dim3 dimBlock  = dim3(16, 16, 1);
  static dim3 dimGrid = dim3(iDivUp(projections_size[0], dimBlock.x), iDivUp(projections_size[1], dimBlock.x));

  // Reset projection
  cudaMemset((void *)dev_proj, 0, projections_size[0]*projections_size[1]*sizeof(float) );
  CUDA_CHECK_ERROR;

  // mu matrix
  float *dev_mu;
  cudaMalloc( (void**)&dev_mu, 256*sizeof(float) );
  cudaMemcpy (dev_mu, mu, 256*sizeof(float), cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR;
  cudaBindTexture (0, tex_mu, dev_mu, 256*sizeof(float) );
  CUDA_CHECK_ERROR;

  // Copy matrix and bind data to the texture
  float *dev_matrix;
  cudaMalloc( (void**)&dev_matrix, 12*sizeof(float) );
  cudaMemcpy (dev_matrix, matrix, 12*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTexture (0, tex_matrix, dev_matrix, 12*sizeof(float) );
  CUDA_CHECK_ERROR;

  // Converting arrays to CUDA format and setting parameters
  float3 sourcePos;
  sourcePos.x = (float) source_position[0];
  sourcePos.y = (float) source_position[1];
  sourcePos.z = (float) source_position[2];
  int2 projSize;
  projSize.x = projections_size[0];
  projSize.y = projections_size[1];
  float3 boxMin;
  boxMin.x = box_min[0];
  boxMin.y = box_min[1];
  boxMin.z = box_min[2];
  float3 boxMax;
  boxMax.x = box_max[0];
  boxMax.y = box_max[1];
  boxMax.z = box_max[2];
  float3 dev_spacing;
  dev_spacing.x = spacing[0];
  dev_spacing.y = spacing[1];
  dev_spacing.z = spacing[2];

  // Calling kernel
  kernel_forwardProject <<< dimGrid, dimBlock >>> (dev_proj, sourcePos, projSize, t_step, boxMin, boxMax, dev_spacing);
  cudaDeviceSynchronize();

  CUDA_CHECK_ERROR;

  // Unbind the volume and matrix textures
  cudaUnbindTexture (tex_vol);
  CUDA_CHECK_ERROR;
  cudaUnbindTexture (tex_matrix);
  CUDA_CHECK_ERROR;
  cudaUnbindTexture (tex_mu);
  CUDA_CHECK_ERROR;

  // Cleanup
  cudaFreeArray ((cudaArray*)array_vol);
  CUDA_CHECK_ERROR;
  cudaFree (dev_matrix);
  CUDA_CHECK_ERROR;
  cudaFree (dev_mu);
  CUDA_CHECK_ERROR;
}
