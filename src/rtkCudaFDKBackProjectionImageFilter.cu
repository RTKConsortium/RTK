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

/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/*****************
*  rtk #includes *
*****************/
#include "rtkCudaUtilities.hcu"
#include "rtkConfiguration.h"
#include "rtkCudaFDKBackProjectionImageFilter.hcu"

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

// T E X T U R E S ////////////////////////////////////////////////////////
texture<float, cudaTextureType2DLayered> tex_proj;
texture<float, 3, cudaReadModeElementType> tex_proj_3D;

// Constant memory
__constant__ float c_matrices[SLAB_SIZE * 12]; //Can process stacks of at most SLAB_SIZE projections
__constant__ int3 c_projSize;
__constant__ int3 c_vol_size;

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__global__
void kernel_fdk(float *dev_vol_in, float *dev_vol_out, unsigned int Blocks_Y)
{
  // CUDA 2.0 does not allow for a 3D grid, which severely
  // limits the manipulation of large 3D arrays of data.  The
  // following code is a hack to bypass this implementation
  // limitation.
  unsigned int blockIdx_z = blockIdx.y / Blocks_Y;
  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

  if (i >= c_vol_size.x || j >= c_vol_size.y || k >= c_vol_size.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*c_vol_size.y)*(c_vol_size.x);

  float3 ip;
  float  voxel_data = 0;

  for (unsigned int proj = 0; proj<c_projSize.z; proj++)
    {
    // matrix multiply
    ip = matrix_multiply(make_float3(i,j,k), &(c_matrices[12*proj]));

    // Change coordinate systems
    ip.z = 1 / ip.z;
    ip.x = ip.x * ip.z;
    ip.y = ip.y * ip.z;

    // Get texture point, clip left to GPU
    voxel_data += tex3D(tex_proj_3D, ip.x, ip.y, proj + 0.5) *  ip.z * ip.z;
    }

  // Place it into the volume
  dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel_data;
}

__global__
void kernel_fdk_3Dgrid(float *dev_vol_in, float * dev_vol_out)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= c_vol_size.x || j >= c_vol_size.y || k >= c_vol_size.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*c_vol_size.y)*(c_vol_size.x);

  float3 ip;
  float  voxel_data = 0;

  for (unsigned int proj = 0; proj<c_projSize.z; proj++)
    {
    // matrix multiply
    ip = matrix_multiply(make_float3(i,j,k), &(c_matrices[12*proj]));

    // Change coordinate systems
    ip.z = 1 / ip.z;
    ip.x = ip.x * ip.z;
    ip.y = ip.y * ip.z;

    // Get texture point, clip left to GPU, and accumulate in voxel_data
    voxel_data += tex2DLayered(tex_proj, ip.x, ip.y, proj) *  ip.z * ip.z;
    }

  // Place it into the volume
  dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel_data;
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_back_project /////////////////////////////
void
CUDA_reconstruct_conebeam(
  int proj_size[3],
  int vol_size[3],
  float *matrices,
  float *dev_vol_in,
  float *dev_vol_out,
  float *dev_proj)
{
  int device;
  cudaGetDevice(&device);

  // Copy the size of inputs into constant memory
  cudaMemcpyToSymbol(c_projSize, proj_size, sizeof(int3));
  cudaMemcpyToSymbol(c_vol_size, vol_size, sizeof(int3));

  // Copy the projection matrices into constant memory
  cudaMemcpyToSymbol(c_matrices, &(matrices[0]), 12 * sizeof(float) * proj_size[2]);

  // set texture parameters
  tex_proj.addressMode[0] = cudaAddressModeBorder;
  tex_proj.addressMode[1] = cudaAddressModeBorder;
  tex_proj.addressMode[2] = cudaAddressModeBorder;
  tex_proj.filterMode = cudaFilterModeLinear;
  tex_proj.normalized = false; // don't access with normalized texture coords

  tex_proj_3D.addressMode[0] = cudaAddressModeBorder;
  tex_proj_3D.addressMode[1] = cudaAddressModeBorder;
  tex_proj_3D.addressMode[2] = cudaAddressModeBorder;
  tex_proj_3D.filterMode = cudaFilterModeLinear;
  tex_proj_3D.normalized = false; // don't access with normalized texture coords

  // Copy projection data to array, bind the array to the texture
  cudaExtent projExtent = make_cudaExtent(proj_size[0], proj_size[1], proj_size[2]);
  cudaArray *array_proj;
  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CUDA_CHECK_ERROR;

  // Allocate array for input projections, in order to bind them to
  // either a 2D layered texture (requires GetCudaComputeCapability >= 2.0) or
  // a 3D texture
  if(CUDA_VERSION<4000 || GetCudaComputeCapability(device).first<=1)
    cudaMalloc3DArray((cudaArray**)&array_proj, &channelDesc, projExtent);
  else
    cudaMalloc3DArray((cudaArray**)&array_proj, &channelDesc, projExtent, cudaArrayLayered);
  CUDA_CHECK_ERROR;

  // Copy data to 3D array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr   = make_cudaPitchedPtr(dev_proj, proj_size[0]*sizeof(float), proj_size[0], proj_size[1]);
  copyParams.dstArray = (cudaArray*)array_proj;
  copyParams.extent   = projExtent;
  copyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&copyParams);
  CUDA_CHECK_ERROR;

  // Thread Block Dimensions
  const int tBlock_x = 16;
  const int tBlock_y = 4;
  const int tBlock_z = 4;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int  blocksInX = (vol_size[0]-1)/tBlock_x + 1;
  unsigned int  blocksInY = (vol_size[1]-1)/tBlock_y + 1;
  unsigned int  blocksInZ = (vol_size[2]-1)/tBlock_z + 1;

  // Run kernels. Note: Projection data is passed via texture memory,
  // transform matrix is passed via constant memory
  if(CUDA_VERSION<4000 || GetCudaComputeCapability(device).first<=1)
    {
    // Compute block and grid sizes
    dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

    // Bind the array of projections to a 3D texture
    cudaBindTextureToArray(tex_proj_3D, (cudaArray*)array_proj, channelDesc);
    CUDA_CHECK_ERROR;

    kernel_fdk <<< dimGrid, dimBlock >>> ( dev_vol_in,
                                           dev_vol_out,
                                           blocksInY );

    // Unbind the image and projection matrix textures
    cudaUnbindTexture (tex_proj_3D);
    CUDA_CHECK_ERROR;
    }
  else
    {
    // Compute block and grid sizes
    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
    CUDA_CHECK_ERROR;

    // Bind the array of projections to a 2D layered texture
    cudaBindTextureToArray(tex_proj, (cudaArray*)array_proj, channelDesc);
    CUDA_CHECK_ERROR;

    kernel_fdk_3Dgrid <<< dimGrid, dimBlock >>> ( dev_vol_in,
                                                  dev_vol_out);

    // Unbind the image and projection matrix textures
    cudaUnbindTexture (tex_proj);
    CUDA_CHECK_ERROR;
    }

  // Cleanup
  cudaFreeArray ((cudaArray*)array_proj);
  CUDA_CHECK_ERROR;
}

