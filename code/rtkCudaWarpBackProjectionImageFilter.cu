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
texture<float, 2, cudaReadModeElementType> tex_img;
texture<float, 3, cudaReadModeElementType> tex_xdvf;
texture<float, 3, cudaReadModeElementType> tex_ydvf;
texture<float, 3, cudaReadModeElementType> tex_zdvf;
///////////////////////////////////////////////////////////////////////////

// CONSTANTS //////////////////////////////////////////////////////////////
__constant__ float c_matrix[12];
__constant__ float c_IndexInputToIndexDVFMatrix[12];
__constant__ float c_PPInputToIndexInputMatrix[12];
__constant__ float c_IndexInputToPPInputMatrix[12];
////////////////////////////////////////////////////////////////////////////

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__global__
void kernel_warp_back_project_3Dgrid(float *dev_vol_in,
                                     float * dev_vol_out,
                                     int3 vol_size)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= vol_size.x || j >= vol_size.y || k >= vol_size.z)
    {
    return;
    }

  // Index row major into the volume
  long int vol_idx = i + (j + k*vol_size.y)*(vol_size.x);

  float3 IndexInDVF;
  IndexInDVF.x =  c_IndexInputToIndexDVFMatrix[0] * i
                + c_IndexInputToIndexDVFMatrix[1] * j
                + c_IndexInputToIndexDVFMatrix[2] * k
                + c_IndexInputToIndexDVFMatrix[3];
  IndexInDVF.y =  c_IndexInputToIndexDVFMatrix[4] * i
                + c_IndexInputToIndexDVFMatrix[5] * j
                + c_IndexInputToIndexDVFMatrix[6] * k
                + c_IndexInputToIndexDVFMatrix[7];
  IndexInDVF.z =  c_IndexInputToIndexDVFMatrix[8] * i
                + c_IndexInputToIndexDVFMatrix[9] * j
                + c_IndexInputToIndexDVFMatrix[10] * k
                + c_IndexInputToIndexDVFMatrix[11];

  // Get each component of the displacement vector by
  // interpolation in the dvf
  float3 Displacement;
  Displacement.x = tex3D(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.y = tex3D(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.z = tex3D(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

  float3 PP; //Physical point in input
  PP.x =  c_IndexInputToPPInputMatrix[0] * i
                + c_IndexInputToPPInputMatrix[1] * j
                + c_IndexInputToPPInputMatrix[2] * k
                + c_IndexInputToPPInputMatrix[3];
  PP.y =  c_IndexInputToPPInputMatrix[4] * i
                + c_IndexInputToPPInputMatrix[5] * j
                + c_IndexInputToPPInputMatrix[6] * k
                + c_IndexInputToPPInputMatrix[7];Dear all,
  PP.z =  c_IndexInputToPPInputMatrix[8] * i
                + c_IndexInputToPPInputMatrix[9] * j
                + c_IndexInputToPPInputMatrix[10] * k
                + c_IndexInputToPPInputMatrix[11];

  // Get the index corresponding to the current physical point in output displaced by the displacement vector
  // Overwriting the PP variable (physical point in input) is less readable, but results in a significant performance boost
  PP.x = PP.x + Displacement.x;
  PP.y = PP.y + Displacement.y;
  PP.z = PP.z + Displacement.z;

  float3 IndexInInput;
  IndexInInput.x =  c_PPInputToIndexInputMatrix[0] * PP.x
                  + c_PPInputToIndexInputMatrix[1] * PP.y
                  + c_PPInputToIndexInputMatrix[2] * PP.z
                  + c_PPInputToIndexInputMatrix[3];
  IndexInInput.y =  c_PPInputToIndexInputMatrix[4] * PP.x
                  + c_PPInputToIndexInputMatrix[5] * PP.y
                  + c_PPInputToIndexInputMatrix[6] * PP.z
                  + c_PPInputToIndexInputMatrix[7];
  IndexInInput.z =  c_PPInputToIndexInputMatrix[8] * PP.x
                  + c_PPInputToIndexInputMatrix[9] * PP.y
                  + c_PPInputToIndexInputMatrix[10]* PP.z
                  + c_PPInputToIndexInputMatrix[11];

  float3 ip;
  float  voxel_data;

  // matrix multiply
  ip.x =  c_matrix[0] * IndexInInput.x
        + c_matrix[1] * IndexInInput.y
        + c_matrix[2] * IndexInInput.z
        + c_matrix[3];
  ip.y =  c_matrix[4] * IndexInInput.x
        + c_matrix[5] * IndexInInput.y
        + c_matrix[6] * IndexInInput.z
        + c_matrix[7];
  ip.z =  c_matrix[8] * IndexInInput.x
        + c_matrix[9] * IndexInInput.y
        + c_matrix[10] * IndexInInput.z
        + c_matrix[11];

  // Change coordinate systems
  ip.z = 1 / ip.z;
  ip.x = ip.x * ip.z;
  ip.y = ip.y * ip.z;

  // Get texture point, clip left to GPU
  voxel_data = tex2D(tex_img, ip.x, ip.y);

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
CUDA_warp_back_project(int img_size[2],
  int vol_size[3],
  int dvf_size[3],
  float matrix[12],
  float *dev_vol_in,
  float *dev_vol_out,
  float *dev_img,
  float *dev_input_dvf,
  float IndexInputToIndexDVFMatrix[12],
  float PPInputToIndexInputMatrix[12],
  float IndexInputToPPInputMatrix[12])
{
  // Create CUBLAS context
  cublasHandle_t  handle;
  cublasCreate(&handle);

  // set texture parameters
  tex_img.addressMode[0] = cudaAddressModeBorder;
  tex_img.addressMode[1] = cudaAddressModeBorder;
  tex_img.filterMode = cudaFilterModeLinear;
  tex_img.normalized = false; // don't access with normalized texture coords

  // copy image data to array, bind the array to the texture
  cudaArray *array_img;
  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaMallocArray( &array_img, &channelDesc, img_size[0], img_size[1] );
  cudaMemcpyToArray( array_img, 0, 0, dev_img, img_size[0] * img_size[1] * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaBindTextureToArray( tex_img, (cudaArray*)array_img, channelDesc);

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
  cudaMemcpyToSymbol (c_matrix, matrix, 12*sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (c_IndexInputToIndexDVFMatrix, IndexInputToIndexDVFMatrix, 12*sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (c_PPInputToIndexInputMatrix,  PPInputToIndexInputMatrix,  12*sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (c_IndexInputToPPInputMatrix,  IndexInputToPPInputMatrix,  12*sizeof(float), 0, cudaMemcpyHostToDevice);

  // Thread Block Dimensions
  const int tBlock_x = 16;
  const int tBlock_y = 4;
  const int tBlock_z = 4;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int  blocksInX = (vol_size[0]-1)/tBlock_x + 1;
  unsigned int  blocksInY = (vol_size[1]-1)/tBlock_y + 1;
  unsigned int  blocksInZ = (vol_size[2]-1)/tBlock_z + 1;

  dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

  // Note: cbi->img is passed via texture memory
  // Matrices are passed via constant memory
  //-------------------------------------
  kernel_warp_back_project_3Dgrid <<< dimGrid, dimBlock >>> ( dev_vol_in,
                                                              dev_vol_out,
                                                              make_int3(vol_size[0], vol_size[1], vol_size[2]));
  CUDA_CHECK_ERROR;

  // Unbind the image and projection matrix textures
  cudaUnbindTexture (tex_xdvf);
  cudaUnbindTexture (tex_ydvf);
  cudaUnbindTexture (tex_zdvf);
  cudaUnbindTexture (tex_img);

  // Cleanup
  cudaFreeArray ((cudaArray*) DVFcomponentArrays[0]);
  cudaFreeArray ((cudaArray*) DVFcomponentArrays[1]);
  cudaFreeArray ((cudaArray*) DVFcomponentArrays[2]);
  delete[] DVFcomponentArrays;
  cudaFreeArray ((cudaArray*)array_img);
  CUDA_CHECK_ERROR;

  // Destroy CUBLAS context
  cublasDestroy(handle);
}
