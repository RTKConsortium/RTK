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
#include "rtkMacro.h"

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
//#include "/home/mvila/NVIDIA_GPU_Computing_SDK/C/common/inc/cutil_math.h"
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
///////////////////////////////////////////////////////////////////////////

//__constant__ float3 spacingSquare;  // inverse view matrix

struct Ray {
        float3 o;	// origin
        float3 d;	// direction
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
  pixelPos.x = tex1Dfetch(tex_matrix,  0)*i + tex1Dfetch(tex_matrix,  1)*j +
               tex1Dfetch(tex_matrix,  3);
  pixelPos.y = tex1Dfetch(tex_matrix,  4)*i + tex1Dfetch(tex_matrix,  5)*j +
               tex1Dfetch(tex_matrix,  7);
  pixelPos.z = tex1Dfetch(tex_matrix,  8)*i + tex1Dfetch(tex_matrix,  9)*j +
               tex1Dfetch(tex_matrix, 11);

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
    // Read from 3D texture from volume
    sample = tex3D(tex_vol, pos.x, pos.y, pos.z);
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
// FUNCTION: CUDA_forward_project_init() /////////////////////////////
void
CUDA_forward_project_init(int proj_dim[2],
                          int vol_dim[3],
                          float *&dev_vol,          // Holds voxels on device
                          float *&dev_proj,         // Holds image pixels on device
                          float *&dev_matrix,       // Holds matrix on device
                          const float *host_volume) // Holds volume on host
{
  // Size of volume Malloc
  cudaExtent volSize =  make_cudaExtent(vol_dim[0], vol_dim[1], vol_dim[2]);
  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  // CUDA device pointers
  // Allocate memory for system matrix on the device
  cudaMalloc( (void**)&dev_matrix, 16*sizeof(float) );
  CUDA_CHECK_ERROR;
  // Allocate memory for reconstructed volume on the device
  cudaMalloc3DArray((cudaArray**)&dev_vol, &channelDesc, volSize);  //cudaMallocArray( (cudaArray**)&dev_vol, volSize_bytes);
  CUDA_CHECK_ERROR;
  // Allocate memory for projections on the device
  cudaMalloc((void **)&dev_proj, proj_dim[0]*proj_dim[1]*sizeof(float) );
  CUDA_CHECK_ERROR;

  // CUDA data transfer, host data to the CUDA device
  cudaMemset((void *)dev_proj, 0, proj_dim[0]*proj_dim[1]*sizeof(float) );
  CUDA_CHECK_ERROR;

  //NOTE: cudaMemcpy of the matrix is done afterwards, just before calling kernel.

  // copy data to 3D array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr   = make_cudaPitchedPtr((void*)host_volume, vol_dim[0]*sizeof(float), vol_dim[0], vol_dim[1]);
  copyParams.dstArray = (cudaArray*)dev_vol;
  copyParams.extent   = volSize;
  copyParams.kind     = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&copyParams);
  CUDA_CHECK_ERROR;
  // Set texture parameters
  tex_vol.normalized = false;                      // access with normalized texture coordinates
  tex_vol.filterMode = cudaFilterModeLinear;      // linear interpolation
  tex_vol.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
  tex_vol.addressMode[1] = cudaAddressModeClamp;
  // Bind 3D array to 3D texture
  cudaBindTextureToArray(tex_vol, (cudaArray*)dev_vol, channelDesc);
  CUDA_CHECK_ERROR;
}

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_forward_project() //////////////////////////////////
void
CUDA_forward_project(int blockSize[3],
                     float *host_proj,
                     float *dev_proj,
                     double source_position[3],
                     int projections_size[2],
                     float t_step,
                     float *dev_matrix,
                     float matrix[16],
                     float box_min[3],
                     float box_max[3],
                     float spacing[3])
{
  static dim3 dimBlock  = dim3(blockSize[0], blockSize[1], blockSize[2]);
  static dim3 dimGrid = dim3(iDivUp(projections_size[0], dimBlock.x), iDivUp(projections_size[1], dimBlock.x));

  // Copy matrix and bind data to the texture
  cudaMemcpy (dev_matrix, matrix, 16*sizeof(float), cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR;
  cudaBindTexture (0, tex_matrix, dev_matrix, 16*sizeof(float) );
  CUDA_CHECK_ERROR;
  // Converting arrays to CUDA format and setting parameters
  float3 sourcePos;
  sourcePos.x = source_position[0];
  sourcePos.y = source_position[1];
  sourcePos.z = source_position[2];
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

  // Copy reconstructed volume from device to host
  size_t projSize_bytes = (projections_size[0]*projections_size[1])*sizeof(float);
  cudaMemcpy (host_proj, dev_proj, projSize_bytes, cudaMemcpyDeviceToHost);
  CUDA_CHECK_ERROR;
}

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_forward_project_cleanup() //////////////////////////
void
CUDA_forward_project_cleanup(int proj_dim[2],
                             float *dev_vol,
                             float *dev_proj,
                             float *dev_matrix)
{
  // Unbind the volume and matrix textures
  cudaUnbindTexture (tex_vol);
  CUDA_CHECK_ERROR;
  cudaUnbindTexture (tex_matrix);
  CUDA_CHECK_ERROR;
  // Deallocate memory on the device
  cudaFree (dev_proj);
  CUDA_CHECK_ERROR;
  cudaFree (dev_matrix);
  CUDA_CHECK_ERROR;
  cudaFreeArray ((cudaArray*)dev_vol);
  CUDA_CHECK_ERROR;

}
