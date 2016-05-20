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
#include "rtkCudaWarpImageFilter.hcu"

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
texture<float, 1, cudaReadModeElementType> tex_IndexOutputToPPOutputMatrix;
texture<float, 1, cudaReadModeElementType> tex_IndexOutputToIndexDVFMatrix;
texture<float, 1, cudaReadModeElementType> tex_PPInputToIndexInputMatrix;

texture<float, 3, cudaReadModeElementType> tex_xdvf;
texture<float, 3, cudaReadModeElementType> tex_ydvf;
texture<float, 3, cudaReadModeElementType> tex_zdvf;
texture<float, 3, cudaReadModeElementType> tex_input_vol;
///////////////////////////////////////////////////////////////////////////

// CONSTANTS //////////////////////////////////////////////////////////////
__constant__ float c_IndexOutputToPPOutputMatrix[12];
__constant__ float c_IndexOutputToIndexDVFMatrix[12];
__constant__ float c_PPInputToIndexInputMatrix[12];
////////////////////////////////////////////////////////////////////////////

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__global__
void kernel(float *dev_vol_out, int3 vol_dim, unsigned int Blocks_Y)
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

  if (i >= vol_dim.x || j >= vol_dim.y || k >= vol_dim.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*vol_dim.y)*(vol_dim.x);

  // Matrix multiply to get the index in the DVF texture of the current point in the output volume
  float3 IndexInDVF = matrix_multiply(make_float3(i,j,k), c_IndexOutputToIndexDVFMatrix);

  // Get each component of the displacement vector by
  // interpolation in the dvf
  float3 Displacement;
  Displacement.x = tex3D(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.y = tex3D(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.z = tex3D(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

  // Matrix multiply to get the physical coordinates of the current point in the output volume
  float3 PPinOutput = matrix_multiply(make_float3(i,j,k), c_IndexOutputToPPOutputMatrix);

  // Get the index corresponding to the current physical point in output displaced by the displacement vector
  float3 PPDisplaced;
  PPDisplaced.x = PPinOutput.x + Displacement.x;
  PPDisplaced.y = PPinOutput.y + Displacement.y;
  PPDisplaced.z = PPinOutput.z + Displacement.z;
  float3 IndexInInput = matrix_multiply(PPDisplaced, c_PPInputToIndexInputMatrix);

  // Interpolate in the input and copy into the output
  dev_vol_out[vol_idx] = tex3D(tex_input_vol, IndexInInput.x + 0.5f, IndexInInput.y + 0.5f, IndexInInput.z + 0.5f);
}

__global__
void kernel_3Dgrid(float * dev_vol_out, int3 vol_dim)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= vol_dim.x || j >= vol_dim.y || k >= vol_dim.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*vol_dim.y)*(vol_dim.x);

  // Matrix multiply to get the index in the DVF texture of the current point in the output volume
  float3 IndexInDVF = matrix_multiply(make_float3(i,j,k), c_IndexOutputToIndexDVFMatrix);

  // Get each component of the displacement vector by
  // interpolation in the dvf
  float3 Displacement;
  Displacement.x = tex3D(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.y = tex3D(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.z = tex3D(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

  // Matrix multiply to get the physical coordinates of the current point in the output volume
  float3 PP = matrix_multiply(make_float3(i,j,k), c_IndexOutputToPPOutputMatrix);

  // Get the index corresponding to the current physical point in output displaced by the displacement vector
  PP += Displacement;

  // Convert it to a continuous index
  float3 IndexInInput = matrix_multiply(PP, c_PPInputToIndexInputMatrix);

  // Interpolate in the input and copy into the output
  dev_vol_out[vol_idx] = tex3D(tex_input_vol, IndexInInput.x + 0.5f, IndexInInput.y + 0.5f, IndexInInput.z + 0.5f);
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_warp /////////////////////////////
void
CUDA_warp(int input_vol_dim[3],
    int input_dvf_dim[3],
    int output_vol_dim[3],
    float IndexOutputToPPOutputMatrix[12],
    float IndexOutputToIndexDVFMatrix[12],
    float PPInputToIndexInputMatrix[12],
    float *dev_input_vol,
    float *dev_output_vol,
    float *dev_DVF,
    bool isLinear)
{

  // Prepare channel description for arrays
  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  // Create CUBLAS context
  cublasHandle_t  handle;
  cublasCreate(&handle);

  ///////////////////////////////////
  // For each component of the dvf, perform a strided copy (pick every third
  // float from dev_input_dvf) into a 3D array, and bind the array to a 3D texture

  // Extent stuff, will be used for each component extraction
  cudaExtent dvfExtent = make_cudaExtent(input_dvf_dim[0], input_dvf_dim[1], input_dvf_dim[2]);

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
  int numel = input_dvf_dim[0] * input_dvf_dim[1] * input_dvf_dim[2];
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
    float * pComponent = dev_DVF + component;
    cublasSaxpy(handle, numel, &one, pComponent, 3, DVFcomponent, 1);

    // Allocate the cudaArray and fill it with the current DVFcomponent
    cudaMalloc3DArray((cudaArray**)& DVFcomponentArrays[component], &channelDesc, dvfExtent);
    cudaMemcpy3DParms CopyParams = {0};
    CopyParams.srcPtr   = make_cudaPitchedPtr(DVFcomponent, input_dvf_dim[0] * sizeof(float), input_dvf_dim[0], input_dvf_dim[1]);
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

  ///////////////////////////////////
  // Do the same for the input volume

  // Extent stuff
  cudaExtent volExtent = make_cudaExtent(input_vol_dim[0], input_vol_dim[1], input_vol_dim[2]);

  // Set texture parameters
  tex_input_vol.addressMode[0] = cudaAddressModeBorder;
  tex_input_vol.addressMode[1] = cudaAddressModeBorder;
  tex_input_vol.addressMode[2] = cudaAddressModeBorder;
  tex_input_vol.normalized = false; // don't access with normalized texture coords
  if (isLinear)
    tex_input_vol.filterMode = cudaFilterModeLinear;
  else
    tex_input_vol.filterMode = cudaFilterModePoint;

  // Allocate the array
  cudaArray *array_input_vol;
  cudaMalloc3DArray((cudaArray**)&array_input_vol, &channelDesc, volExtent);
  CUDA_CHECK_ERROR;

  // Copy image data to array
  cudaMemcpy3DParms inputCopyParams = {0};
  inputCopyParams.srcPtr   = make_cudaPitchedPtr(dev_input_vol, input_vol_dim[0]*sizeof(float), input_vol_dim[0], input_vol_dim[1]);
  inputCopyParams.dstArray = (cudaArray*)array_input_vol;
  inputCopyParams.extent   = volExtent;
  inputCopyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&inputCopyParams);
  CUDA_CHECK_ERROR;

  // Bind 3D arrays to 3D textures
  cudaBindTextureToArray(tex_input_vol, (cudaArray*)array_input_vol, channelDesc);
  CUDA_CHECK_ERROR;

  // Copy matrices into constant memory
  cudaMemcpyToSymbol (c_IndexOutputToPPOutputMatrix, IndexOutputToPPOutputMatrix, 12*sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (c_IndexOutputToIndexDVFMatrix,  IndexOutputToIndexDVFMatrix,  12*sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (c_PPInputToIndexInputMatrix,  PPInputToIndexInputMatrix,  12*sizeof(float), 0, cudaMemcpyHostToDevice);

  //////////////////////////////////////
  /// Run

  int device;
  cudaGetDevice(&device);

  // Thread Block Dimensions
  const int tBlock_x = 16;
  const int tBlock_y = 4;
  const int tBlock_z = 4;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int  blocksInX = (output_vol_dim[0]-1)/tBlock_x + 1;
  unsigned int  blocksInY = (output_vol_dim[1]-1)/tBlock_y + 1;
  unsigned int  blocksInZ = (output_vol_dim[2]-1)/tBlock_z + 1;

  if(CUDA_VERSION<4000 || GetCudaComputeCapability(device).first<=1)
    {
    dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

    // Note: the DVF and input image are passed via texture memory
    //-------------------------------------
    kernel <<< dimGrid, dimBlock >>> ( dev_output_vol,
                                       make_int3(output_vol_dim[0], output_vol_dim[1], output_vol_dim[2]),
                                       blocksInY );
    }
  else
    {
    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

    // Note: the DVF and input image are passed via texture memory
    //-------------------------------------
    kernel_3Dgrid <<< dimGrid, dimBlock >>> ( dev_output_vol,
                                              make_int3(output_vol_dim[0], output_vol_dim[1], output_vol_dim[2]));
    }

  CUDA_CHECK_ERROR;

  // Unbind the image and projection matrix textures
  cudaUnbindTexture (tex_xdvf);
  cudaUnbindTexture (tex_ydvf);
  cudaUnbindTexture (tex_zdvf);
  cudaUnbindTexture (tex_input_vol);
  CUDA_CHECK_ERROR;

  // Cleanup
  CUDA_CHECK_ERROR;
  cudaFreeArray ((cudaArray*) DVFcomponentArrays[0]);
  cudaFreeArray ((cudaArray*) DVFcomponentArrays[1]);
  cudaFreeArray ((cudaArray*) DVFcomponentArrays[2]);
  cudaFreeArray ((cudaArray*) array_input_vol);
  delete[] DVFcomponentArrays;
  CUDA_CHECK_ERROR;

  // Destroy CUBLAS context
  cublasDestroy(handle);
}
