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
#include "rtkCudaWarpBackProjectionImageFilter.hcu"

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

// T E X T U R E S ////////////////////////////////////////////////////////
texture<float, cudaTextureType2DLayered> tex_proj;
texture<float, 3, cudaReadModeElementType> tex_proj_3D;

texture<float, 3, cudaReadModeElementType> tex_xdvf;
texture<float, 3, cudaReadModeElementType> tex_ydvf;
texture<float, 3, cudaReadModeElementType> tex_zdvf;
///////////////////////////////////////////////////////////////////////////

// CONSTANTS //////////////////////////////////////////////////////////////
__constant__ float c_matrices[SLAB_SIZE * 12]; //Can process stacks of at most SLAB_SIZE projections
__constant__ float c_volIndexToProjPP[SLAB_SIZE * 12];
__constant__ float c_projPPToProjIndex[9];
__constant__ int3 c_projSize;
__constant__ int3 c_volSize;
__constant__ float c_IndexInputToIndexDVFMatrix[12];
__constant__ float c_PPInputToIndexInputMatrix[12];
__constant__ float c_IndexInputToPPInputMatrix[12];
////////////////////////////////////////////////////////////////////////////

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__global__
void kernel_warp_back_project(float *dev_vol_in, float * dev_vol_out, unsigned int Blocks_Y)
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

  if (i >= c_volSize.x || j >= c_volSize.y || k >= c_volSize.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*c_volSize.y)*(c_volSize.x);

  float3 IndexInDVF, Displacement, PP, IndexInInput, ip;
  float  voxel_data = 0;

  for (unsigned int proj = 0; proj<c_projSize.z; proj++)
    {
    // Compute the index in the DVF
    IndexInDVF = matrix_multiply(make_float3(i, j, k),  c_IndexInputToIndexDVFMatrix);

    // Get each component of the displacement vector by interpolation in the DVF
    Displacement.x = tex3D(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
    Displacement.y = tex3D(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
    Displacement.z = tex3D(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

    // Compute the physical point in input + the displacement vector
    PP = matrix_multiply(make_float3(i, j, k),  c_IndexInputToPPInputMatrix) + Displacement;

    // Convert it to a continuous index
    IndexInInput = matrix_multiply(PP,  c_PPInputToIndexInputMatrix);

    // Project the voxel onto the detector to find out which value to add to it
    ip = matrix_multiply(IndexInInput, &(c_matrices[12*proj]));;

    // Change coordinate systems
    ip.z = 1 / ip.z;
    ip.x = ip.x * ip.z;
    ip.y = ip.y * ip.z;

    // Get texture point, clip left to GPU
    voxel_data += tex3D(tex_proj_3D, ip.x, ip.y, proj + 0.5);
    }

  // Place it into the volume
  dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel_data;
}

__global__
void kernel_warp_back_project_cylindrical_detector(float *dev_vol_in, float * dev_vol_out, unsigned int Blocks_Y, double radius)
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

  if (i >= c_volSize.x || j >= c_volSize.y || k >= c_volSize.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*c_volSize.y)*(c_volSize.x);

  float3 IndexInDVF, Displacement, PP, IndexInInput, ip, pp;
  float  voxel_data = 0;

  for (unsigned int proj = 0; proj<c_projSize.z; proj++)
    {
    // Compute the index in the DVF
    IndexInDVF = matrix_multiply(make_float3(i, j, k),  c_IndexInputToIndexDVFMatrix);

    // Get each component of the displacement vector by interpolation in the DVF
    Displacement.x = tex3D(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
    Displacement.y = tex3D(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
    Displacement.z = tex3D(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

    // Compute the physical point in input + the displacement vector
    PP = matrix_multiply(make_float3(i, j, k),  c_IndexInputToPPInputMatrix) + Displacement;

    // Convert it to a continuous index
    IndexInInput = matrix_multiply(PP,  c_PPInputToIndexInputMatrix);

    // Project the voxel onto the detector to find out which value to add to it
    pp = matrix_multiply(IndexInInput, &(c_volIndexToProjPP[12*proj]));

    // Change coordinate systems
    pp.z = 1 / pp.z;
    pp.x = pp.x * pp.z;
    pp.y = pp.y * pp.z;

    // Apply correction for cylindrical detector
    double u = pp.x;
    pp.x = radius * atan(u / radius);
    pp.y = pp.y * radius / sqrt(radius * radius + u * u);

    // Get projection index
    ip.x = c_projPPToProjIndex[0 * 3 + 0] * pp.x + c_projPPToProjIndex[0 * 3 + 1] * pp.y + c_projPPToProjIndex[0 * 3 + 2];
    ip.y = c_projPPToProjIndex[1 * 3 + 0] * pp.x + c_projPPToProjIndex[1 * 3 + 1] * pp.y + c_projPPToProjIndex[1 * 3 + 2];

    // Get texture point, clip left to GPU
    voxel_data += tex3D(tex_proj_3D, ip.x, ip.y, proj + 0.5);
    }

  // Place it into the volume
  dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel_data;
}

__global__
void kernel_warp_back_project_3Dgrid(float *dev_vol_in, float * dev_vol_out)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= c_volSize.x || j >= c_volSize.y || k >= c_volSize.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*c_volSize.y)*(c_volSize.x);

  float3 IndexInDVF, Displacement, PP, IndexInInput, ip;
  float  voxel_data = 0;

  for (unsigned int proj = 0; proj<c_projSize.z; proj++)
    {
    // Compute the index in the DVF
    IndexInDVF = matrix_multiply(make_float3(i, j, k),  c_IndexInputToIndexDVFMatrix);

    // Get each component of the displacement vector by interpolation in the DVF
    Displacement.x = tex3D(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
    Displacement.y = tex3D(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
    Displacement.z = tex3D(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

    // Compute the physical point in input + the displacement vector
    PP = matrix_multiply(make_float3(i, j, k),  c_IndexInputToPPInputMatrix) + Displacement;

    // Convert it to a continuous index
    IndexInInput = matrix_multiply(PP,  c_PPInputToIndexInputMatrix);

    // Project the voxel onto the detector to find out which value to add to it
    ip = matrix_multiply(IndexInInput, &(c_matrices[12*proj]));;

    // Change coordinate systems
    ip.z = 1 / ip.z;
    ip.x = ip.x * ip.z;
    ip.y = ip.y * ip.z;

    // Get texture point, clip left to GPU
    voxel_data += tex2DLayered(tex_proj, ip.x, ip.y, proj);
    }

  // Place it into the volume
  dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel_data;
}

__global__
void kernel_warp_back_project_3Dgrid_cylindrical_detector(float *dev_vol_in, float * dev_vol_out, double radius)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= c_volSize.x || j >= c_volSize.y || k >= c_volSize.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*c_volSize.y)*(c_volSize.x);

  float3 IndexInDVF, Displacement, PP, IndexInInput, ip, pp;
  float  voxel_data = 0;

  for (unsigned int proj = 0; proj<c_projSize.z; proj++)
    {
    // Compute the index in the DVF
    IndexInDVF = matrix_multiply(make_float3(i, j, k),  c_IndexInputToIndexDVFMatrix);

    // Get each component of the displacement vector by interpolation in the DVF
    Displacement.x = tex3D(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
    Displacement.y = tex3D(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
    Displacement.z = tex3D(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

    // Compute the physical point in input + the displacement vector
    PP = matrix_multiply(make_float3(i, j, k),  c_IndexInputToPPInputMatrix) + Displacement;

    // Convert it to a continuous index
    IndexInInput = matrix_multiply(PP,  c_PPInputToIndexInputMatrix);

    // Project the voxel onto the detector to find out which value to add to it
    pp = matrix_multiply(IndexInInput, &(c_volIndexToProjPP[12*proj]));

    // Change coordinate systems
    pp.z = 1 / pp.z;
    pp.x = pp.x * pp.z;
    pp.y = pp.y * pp.z;

    // Apply correction for cylindrical detector
    double u = pp.x;
    pp.x = radius * atan(u / radius);
    pp.y = pp.y * radius / sqrt(radius * radius + u * u);

    // Get projection index
    ip.x = c_projPPToProjIndex[0 * 3 + 0] * pp.x + c_projPPToProjIndex[0 * 3 + 1] * pp.y + c_projPPToProjIndex[0 * 3 + 2];
    ip.y = c_projPPToProjIndex[1 * 3 + 0] * pp.x + c_projPPToProjIndex[1 * 3 + 1] * pp.y + c_projPPToProjIndex[1 * 3 + 2];

    // Get texture point, clip left to GPU
    voxel_data += tex2DLayered(tex_proj, ip.x, ip.y, proj);
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
CUDA_warp_back_project(int projSize[3],
  int volSize[3],
  int dvf_size[3],
  float *matrices,
  float *volIndexToProjPPs,
  float *projPPToProjIndex,
  float *dev_vol_in,
  float *dev_vol_out,
  float *dev_proj,
  float *dev_input_dvf,
  float IndexInputToIndexDVFMatrix[12],
  float PPInputToIndexInputMatrix[12],
  float IndexInputToPPInputMatrix[12],
  double radiusCylindricalDetector)
{
  // Create CUBLAS context
  cublasHandle_t  handle;
  cublasCreate(&handle);

  int device;
  cudaGetDevice(&device);

  // Copy the size of inputs into constant memory
  cudaMemcpyToSymbol(c_projSize, projSize, sizeof(int3));
  cudaMemcpyToSymbol(c_volSize, volSize, sizeof(int3));

  // Copy the projection matrices into constant memory
  cudaMemcpyToSymbol(c_matrices,          &(matrices[0]),          12 * sizeof(float) * projSize[2]);
  cudaMemcpyToSymbol(c_volIndexToProjPP,  &(volIndexToProjPPs[0]), 12 * sizeof(float) * projSize[2]);
  cudaMemcpyToSymbol(c_projPPToProjIndex, &(projPPToProjIndex[0]), 9 * sizeof(float));

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
  cudaExtent projExtent = make_cudaExtent(projSize[0], projSize[1], projSize[2]);
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
  copyParams.srcPtr   = make_cudaPitchedPtr(dev_proj, projSize[0]*sizeof(float), projSize[0], projSize[1]);
  copyParams.dstArray = (cudaArray*)array_proj;
  copyParams.extent   = projExtent;
  copyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&copyParams);
  CUDA_CHECK_ERROR;

  // Extent stuff, will be used for each component extraction
  cudaExtent dvfExtent = make_cudaExtent(dvf_size[0], dvf_size[1], dvf_size[2]);

  // Set texture parameters
  tex_xdvf.addressMode[0] = cudaAddressModeBorder;
  tex_xdvf.addressMode[1] = cudaAddressModeBorder;
  tex_xdvf.addressMode[2] = cudaAddressModeBorder;
  tex_xdvf.filterMode = cudaFilterModeLinear;
  tex_xdvf.normalized = false; // don't access with normalized texture coords

  tex_ydvf.addressMode[0] = cudaAddressModeBorder;
  tex_ydvf.addressMode[1] = cudaAddressModeBorder;
  tex_ydvf.addressMode[2] = cudaAddressModeBorder;
  tex_ydvf.filterMode = cudaFilterModeLinear;
  tex_ydvf.normalized = false;

  tex_zdvf.addressMode[0] = cudaAddressModeBorder;
  tex_zdvf.addressMode[1] = cudaAddressModeBorder;
  tex_zdvf.addressMode[2] = cudaAddressModeBorder;
  tex_zdvf.filterMode = cudaFilterModeLinear;
  tex_zdvf.normalized = false;

  // Allocate an intermediate memory space to extract x, y and z components of the DVF
  float *DVFcomponent;
  int numel = dvf_size[0] * dvf_size[1] * dvf_size[2];
  cudaMalloc(&DVFcomponent, numel * sizeof(float));
  float one = 1.0;

  // Allocate the arrays used for textures
  cudaArray** DVFcomponentArrays = new cudaArray* [3];
  CUDA_CHECK_ERROR;

  // Copy image data to arrays. The tricky part is the make_cudaPitchedPtr.
  // The best way to understand it is to read
  // http://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
  for (unsigned int component = 0; component < 3; component++)
    {
    // Reset the intermediate memory
    cudaMemset((void *)DVFcomponent, 0, numel * sizeof(float));

    // Fill it with the current component
    float * pComponent = dev_input_dvf + component;
    cublasSaxpy(handle, numel, &one, pComponent, 3, DVFcomponent, 1);

    // Allocate the cudaArray and fill it with the current DVFcomponent
    cudaMalloc3DArray((cudaArray**)& DVFcomponentArrays[component], &channelDesc, dvfExtent);
    cudaMemcpy3DParms CopyParams = {0};
    CopyParams.srcPtr   = make_cudaPitchedPtr(DVFcomponent, dvf_size[0] * sizeof(float), dvf_size[0], dvf_size[1]);
    CopyParams.dstArray = (cudaArray*) DVFcomponentArrays[component];
    CopyParams.extent   = dvfExtent;
    CopyParams.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&CopyParams);
    CUDA_CHECK_ERROR;
    }

  // Intermediate memory is no longer needed
  cudaFree (DVFcomponent);

  // Bind 3D arrays to 3D textures
  cudaBindTextureToArray(tex_xdvf, (cudaArray*) DVFcomponentArrays[0], channelDesc);
  cudaBindTextureToArray(tex_ydvf, (cudaArray*) DVFcomponentArrays[1], channelDesc);
  cudaBindTextureToArray(tex_zdvf, (cudaArray*) DVFcomponentArrays[2], channelDesc);
  CUDA_CHECK_ERROR;

  // Copy matrices into constant memory
  cudaMemcpyToSymbol (c_IndexInputToIndexDVFMatrix, IndexInputToIndexDVFMatrix, 12*sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (c_PPInputToIndexInputMatrix,  PPInputToIndexInputMatrix,  12*sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (c_IndexInputToPPInputMatrix,  IndexInputToPPInputMatrix,  12*sizeof(float), 0, cudaMemcpyHostToDevice);

  // Thread Block Dimensions
  const int tBlock_x = 16;
  const int tBlock_y = 4;
  const int tBlock_z = 4;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int  blocksInX = (volSize[0]-1)/tBlock_x + 1;
  unsigned int  blocksInY = (volSize[1]-1)/tBlock_y + 1;
  unsigned int  blocksInZ = (volSize[2]-1)/tBlock_z + 1;

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

    if (radiusCylindricalDetector == 0)
      kernel_warp_back_project <<< dimGrid, dimBlock >>> ( dev_vol_in,
                                                           dev_vol_out,
                                                           blocksInY );
    else
      kernel_warp_back_project_cylindrical_detector <<< dimGrid, dimBlock >>> ( dev_vol_in,
                                                                                dev_vol_out,
                                                                                blocksInY,
                                                                                radiusCylindricalDetector);

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

    // Note: cbi->img is passed via texture memory
    // Matrices are passed via constant memory
    //-------------------------------------
    if (radiusCylindricalDetector == 0)
      kernel_warp_back_project_3Dgrid <<< dimGrid, dimBlock >>> ( dev_vol_in,
                                                                dev_vol_out);
    else
      kernel_warp_back_project_3Dgrid_cylindrical_detector <<< dimGrid, dimBlock >>> ( dev_vol_in,
                                                                                       dev_vol_out,
                                                                                       radiusCylindricalDetector);
    // Unbind the image and projection matrix textures
    cudaUnbindTexture (tex_proj);
    CUDA_CHECK_ERROR;
    }

  // Unbind the image and projection matrix textures
  cudaUnbindTexture (tex_xdvf);
  cudaUnbindTexture (tex_ydvf);
  cudaUnbindTexture (tex_zdvf);

  // Cleanup
  cudaFreeArray ((cudaArray*) DVFcomponentArrays[0]);
  cudaFreeArray ((cudaArray*) DVFcomponentArrays[1]);
  cudaFreeArray ((cudaArray*) DVFcomponentArrays[2]);
  delete[] DVFcomponentArrays;
  cudaFreeArray ((cudaArray*)array_proj);
  CUDA_CHECK_ERROR;

  // Destroy CUBLAS context
  cublasDestroy(handle);
}
