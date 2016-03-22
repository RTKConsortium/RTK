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

// T E X T U R E S ////////////////////////////////////////////////////////
texture<float, 2, cudaReadModeElementType> tex_img;
texture<float, 1, cudaReadModeElementType> tex_matrix;
texture<float, 1, cudaReadModeElementType> tex_IndexInputToPPInputMatrix;
texture<float, 1, cudaReadModeElementType> tex_IndexInputToIndexDVFMatrix;
texture<float, 1, cudaReadModeElementType> tex_PPInputToIndexInputMatrix;

texture<float, 3, cudaReadModeElementType> tex_xdvf;
texture<float, 3, cudaReadModeElementType> tex_ydvf;
texture<float, 3, cudaReadModeElementType> tex_zdvf;
///////////////////////////////////////////////////////////////////////////

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

//__global__
//void kernel(float *dev_vol_in, float *dev_vol_out, int3 vol_size, unsigned int Blocks_Y)
//{
//  // CUDA 2.0 does not allow for a 3D grid, which severely
//  // limits the manipulation of large 3D arrays of data.  The
//  // following code is a hack to bypass this implementation
//  // limitation.
//  unsigned int blockIdx_z = blockIdx.y / Blocks_Y;
//  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
//  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//  unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
//  unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

//  if (i >= vol_size.x || j >= vol_size.y || k >= vol_size.z)
//    {
//    return;
//    }

//  // Index row major into the volume
//  long int vol_idx = i + (j + k*vol_size.y)*(vol_size.x);

//  float3 ip;
//  float  voxel_data;

//  // matrix multiply
//  ip.x = tex1Dfetch(tex_matrix, 0)*i + tex1Dfetch(tex_matrix, 1)*j +
//         tex1Dfetch(tex_matrix, 2)*k + tex1Dfetch(tex_matrix, 3);
//  ip.y = tex1Dfetch(tex_matrix, 4)*i + tex1Dfetch(tex_matrix, 5)*j +
//         tex1Dfetch(tex_matrix, 6)*k + tex1Dfetch(tex_matrix, 7);
//  ip.z = tex1Dfetch(tex_matrix, 8)*i + tex1Dfetch(tex_matrix, 9)*j +
//         tex1Dfetch(tex_matrix, 10)*k + tex1Dfetch(tex_matrix, 11);

//  // Change coordinate systems
//  ip.z = 1 / ip.z;
//  ip.x = ip.x * ip.z;
//  ip.y = ip.y * ip.z;

//  // Get texture point, clip left to GPU
//  voxel_data = tex2D(tex_img, ip.x, ip.y);

//  // Place it into the volume
//  dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel_data;
//}

__global__
void kernel_warp_back_project_3Dgrid(float *dev_vol_in, float * dev_vol_out, int3 vol_size)
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
  IndexInDVF.x =  tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 0) * i
                + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 1) * j
                + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 2) * k
                + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 3);
  IndexInDVF.y =  tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 4) * i
                + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 5) * j
                + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 6) * k
                + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 7);
  IndexInDVF.z =  tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 8) * i
                + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 9) * j
                + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 10) * k
                + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 11);

  // Get each component of the displacement vector by
  // interpolation in the dvf
  float3 Displacement;
  Displacement.x = tex3D(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.y = tex3D(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.z = tex3D(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

  // Matrix multiply to get the physical coordinates of the current point in the output volume
  float3 PPinInput;
  PPinInput.x =  tex1Dfetch(tex_IndexInputToPPInputMatrix, 0) * i
                + tex1Dfetch(tex_IndexInputToPPInputMatrix, 1) * j
                + tex1Dfetch(tex_IndexInputToPPInputMatrix, 2) * k
                + tex1Dfetch(tex_IndexInputToPPInputMatrix, 3);
  PPinInput.y =  tex1Dfetch(tex_IndexInputToPPInputMatrix, 4) * i
                + tex1Dfetch(tex_IndexInputToPPInputMatrix, 5) * j
                + tex1Dfetch(tex_IndexInputToPPInputMatrix, 6) * k
                + tex1Dfetch(tex_IndexInputToPPInputMatrix, 7);
  PPinInput.z =  tex1Dfetch(tex_IndexInputToPPInputMatrix, 8) * i
                + tex1Dfetch(tex_IndexInputToPPInputMatrix, 9) * j
                + tex1Dfetch(tex_IndexInputToPPInputMatrix, 10) * k
                + tex1Dfetch(tex_IndexInputToPPInputMatrix, 11);

  // Get the index corresponding to the current physical point in output displaced by the displacement vector
  float3 PPDisplaced;
  PPDisplaced.x = PPinInput.x + Displacement.x;
  PPDisplaced.y = PPinInput.y + Displacement.y;
  PPDisplaced.z = PPinInput.z + Displacement.z;

  float3 IndexInInput;
  IndexInInput.x =  tex1Dfetch(tex_PPInputToIndexInputMatrix, 0) * PPDisplaced.x
                  + tex1Dfetch(tex_PPInputToIndexInputMatrix, 1) * PPDisplaced.y
                  + tex1Dfetch(tex_PPInputToIndexInputMatrix, 2) * PPDisplaced.z
                  + tex1Dfetch(tex_PPInputToIndexInputMatrix, 3);
  IndexInInput.y =  tex1Dfetch(tex_PPInputToIndexInputMatrix, 4) * PPDisplaced.x
                  + tex1Dfetch(tex_PPInputToIndexInputMatrix, 5) * PPDisplaced.y
                  + tex1Dfetch(tex_PPInputToIndexInputMatrix, 6) * PPDisplaced.z
                  + tex1Dfetch(tex_PPInputToIndexInputMatrix, 7);
  IndexInInput.z =  tex1Dfetch(tex_PPInputToIndexInputMatrix, 8) * PPDisplaced.x
                  + tex1Dfetch(tex_PPInputToIndexInputMatrix, 9) * PPDisplaced.y
                  + tex1Dfetch(tex_PPInputToIndexInputMatrix, 10)* PPDisplaced.z
                  + tex1Dfetch(tex_PPInputToIndexInputMatrix, 11);

  float3 ip;
  float  voxel_data;

  // matrix multiply
  ip.x =  tex1Dfetch(tex_matrix, 0) * IndexInInput.x
        + tex1Dfetch(tex_matrix, 1) * IndexInInput.y
        + tex1Dfetch(tex_matrix, 2) * IndexInInput.z
        + tex1Dfetch(tex_matrix, 3);
  ip.y =  tex1Dfetch(tex_matrix, 4) * IndexInInput.x
        + tex1Dfetch(tex_matrix, 5) * IndexInInput.y
        + tex1Dfetch(tex_matrix, 6) * IndexInInput.z
        + tex1Dfetch(tex_matrix, 7);
  ip.z =  tex1Dfetch(tex_matrix, 8) * IndexInInput.x
        + tex1Dfetch(tex_matrix, 9) * IndexInInput.y
        + tex1Dfetch(tex_matrix, 10) * IndexInInput.z
        + tex1Dfetch(tex_matrix, 11);

  // Change coordinate systems
  ip.z = 1 / ip.z;
  ip.x = ip.x * ip.z;
  ip.y = ip.y * ip.z;

  // Get texture point, clip left to GPU
  voxel_data = tex2D(tex_img, ip.x, ip.y);

  // Place it into the volume
  dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + voxel_data;
}

//__global__
//void kernel_optim(float *dev_vol_in, float *dev_vol_out, int3 vol_size)
//{
//  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  unsigned int j = 0;
//  unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;

//  if (i >= vol_size.x || k >= vol_size.z)
//    {
//    return;
//    }

//  // Index row major into the volume
//  long int vol_idx = i + k*vol_size.y*vol_size.x;

//  float3 ip;

//  // matrix multiply
//  ip.x = tex1Dfetch(tex_matrix, 0)*i + tex1Dfetch(tex_matrix, 2)*k + tex1Dfetch(tex_matrix, 3);
//  ip.y = tex1Dfetch(tex_matrix, 4)*i + tex1Dfetch(tex_matrix, 6)*k + tex1Dfetch(tex_matrix, 7);
//  ip.z = tex1Dfetch(tex_matrix, 8)*i + tex1Dfetch(tex_matrix, 10)*k + tex1Dfetch(tex_matrix, 11);

//  // Change coordinate systems
//  ip.z = 1 / ip.z;
//  ip.x = ip.x * ip.z;
//  ip.y = ip.y * ip.z;
//  float dx = tex1Dfetch(tex_matrix, 1)*ip.z;
//  float dy = tex1Dfetch(tex_matrix, 5)*ip.z;

//  // Place it into the volume segment
//  for(; j<vol_size.y; j++)
//    {
//    dev_vol_out[vol_idx] = dev_vol_in[vol_idx] + tex2D(tex_img, ip.x, ip.y);
//    vol_idx+=vol_size.x;
//    ip.x+=dx;
//    ip.y+=dy;
//    }
//}

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
  float *dev_input_xdvf,
  float *dev_input_ydvf,
  float *dev_input_zdvf,
  float IndexInputToIndexDVFMatrix[12],
  float PPInputToIndexInputMatrix[12],
  float IndexInputToPPInputMatrix[12])
{
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

  // copy matrix, bind data to the texture
  float *dev_matrix;
  cudaMalloc( (void**)&dev_matrix, 12*sizeof(float) );
  cudaMemcpy (dev_matrix, matrix, 12*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTexture (0, tex_matrix, dev_matrix, 12*sizeof(float) );


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

  // Allocate the arrays
  cudaArray *array_xdvf;
  cudaArray *array_ydvf;
  cudaArray *array_zdvf;
  cudaMalloc3DArray((cudaArray**)&array_xdvf, &channelDesc, dvfExtent);
  cudaMalloc3DArray((cudaArray**)&array_ydvf, &channelDesc, dvfExtent);
  cudaMalloc3DArray((cudaArray**)&array_zdvf, &channelDesc, dvfExtent);
  CUDA_CHECK_ERROR;

  // Copy image data to arrays. The tricky part is the make_cudaPitchedPtr.
  // The best way to understand it is to read
  // http://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
  cudaMemcpy3DParms xCopyParams = {0};
  xCopyParams.srcPtr   = make_cudaPitchedPtr(dev_input_xdvf, dvf_size[0] * sizeof(float), dvf_size[0], dvf_size[1]);
  xCopyParams.dstArray = (cudaArray*)array_xdvf;
  xCopyParams.extent   = dvfExtent;
  xCopyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&xCopyParams);
  CUDA_CHECK_ERROR;

  cudaMemcpy3DParms yCopyParams = {0};
  yCopyParams.srcPtr   = make_cudaPitchedPtr(dev_input_ydvf, dvf_size[0] * sizeof(float), dvf_size[0], dvf_size[1]);
  yCopyParams.dstArray = (cudaArray*)array_ydvf;
  yCopyParams.extent   = dvfExtent;
  yCopyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&yCopyParams);
  CUDA_CHECK_ERROR;

  cudaMemcpy3DParms zCopyParams = {0};
  zCopyParams.srcPtr   = make_cudaPitchedPtr(dev_input_zdvf, dvf_size[0] * sizeof(float), dvf_size[0], dvf_size[1]);
  zCopyParams.dstArray = (cudaArray*)array_zdvf;
  zCopyParams.extent   = dvfExtent;
  zCopyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&zCopyParams);
  CUDA_CHECK_ERROR;

  // Bind 3D arrays to 3D textures
  cudaBindTextureToArray(tex_xdvf, (cudaArray*)array_xdvf, channelDesc);
  cudaBindTextureToArray(tex_ydvf, (cudaArray*)array_ydvf, channelDesc);
  cudaBindTextureToArray(tex_zdvf, (cudaArray*)array_zdvf, channelDesc);
  CUDA_CHECK_ERROR;

  ///////////////////////////////////////
  // Copy matrices, bind them to textures

  float *dev_IndexInputToPPInput;
  cudaMalloc( (void**)&dev_IndexInputToPPInput, 12*sizeof(float) );
  cudaMemcpy (dev_IndexInputToPPInput, IndexInputToPPInputMatrix, 12*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTexture (0, tex_IndexInputToPPInputMatrix, dev_IndexInputToPPInput, 12*sizeof(float) );

  float *dev_IndexInputToIndexDVF;
  cudaMalloc( (void**)&dev_IndexInputToIndexDVF, 12*sizeof(float) );
  cudaMemcpy (dev_IndexInputToIndexDVF, IndexInputToIndexDVFMatrix, 12*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTexture (0, tex_IndexInputToIndexDVFMatrix, dev_IndexInputToIndexDVF, 12*sizeof(float) );

  float *dev_PPInputToIndexInput;
  cudaMalloc( (void**)&dev_PPInputToIndexInput, 12*sizeof(float) );
  cudaMemcpy (dev_PPInputToIndexInput, PPInputToIndexInputMatrix, 12*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTexture (0, tex_PPInputToIndexInputMatrix, dev_PPInputToIndexInput, 12*sizeof(float) );



  // The optimized version runs when only one of the axis of the detector is
  // parallel to the y axis of the volume

//  if(fabs(matrix[5])<1e-10 && fabs(matrix[9])<1e-10)
//    {
//    // Thread Block Dimensions
//    const int tBlock_x = 32;
//    const int tBlock_y = 16;

//    // Each segment gets 1 thread
//    unsigned int  blocksInX = (vol_size[0]-1)/tBlock_x + 1;
//    unsigned int  blocksInY = (vol_size[2]-1)/tBlock_y + 1;
//    dim3 dimGrid  = dim3(blocksInX, blocksInY);
//    dim3 dimBlock = dim3(tBlock_x, tBlock_y, 1);

//    // Note: cbi->img AND cbi->matrix are passed via texture memory
//    //-------------------------------------
//    kernel_optim <<< dimGrid, dimBlock >>> ( dev_vol_in,
//                                             dev_vol_out,
//                                             make_int3(vol_size[0], vol_size[1], vol_size[2]) );
//    }
//  else
//    {
//    int device;
//    cudaGetDevice(&device);

    // Thread Block Dimensions
    const int tBlock_x = 16;
    const int tBlock_y = 4;
    const int tBlock_z = 4;

    // Each element in the volume (each voxel) gets 1 thread
    unsigned int  blocksInX = (vol_size[0]-1)/tBlock_x + 1;
    unsigned int  blocksInY = (vol_size[1]-1)/tBlock_y + 1;
    unsigned int  blocksInZ = (vol_size[2]-1)/tBlock_z + 1;

//    if(CUDA_VERSION<4000 || GetCudaComputeCapability(device).first<=1)
//      {
//      dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
//      dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);


//      // Note: cbi->img AND cbi->matrix are passed via texture memory
//      //-------------------------------------
//      kernel <<< dimGrid, dimBlock >>> ( dev_vol_in,
//                                         dev_vol_out,
//                                         make_int3(vol_size[0], vol_size[1], vol_size[2]),
//                                         blocksInY );
//      }
//    else
//      {
      dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
      dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);


      // Note: cbi->img AND cbi->matrix are passed via texture memory
      //-------------------------------------
      kernel_warp_back_project_3Dgrid <<< dimGrid, dimBlock >>> ( dev_vol_in,
                                                dev_vol_out,
                                                make_int3(vol_size[0], vol_size[1], vol_size[2]));
//      }

//    }
  CUDA_CHECK_ERROR;

  // Unbind the image and projection matrix textures
  cudaUnbindTexture (tex_xdvf);
  cudaUnbindTexture (tex_ydvf);
  cudaUnbindTexture (tex_zdvf);
  CUDA_CHECK_ERROR;
  cudaUnbindTexture (tex_img);
  CUDA_CHECK_ERROR;
  cudaUnbindTexture (tex_matrix);
  CUDA_CHECK_ERROR;
  cudaUnbindTexture (tex_IndexInputToPPInputMatrix);
  cudaUnbindTexture (tex_IndexInputToIndexDVFMatrix);
  cudaUnbindTexture (tex_PPInputToIndexInputMatrix);
  CUDA_CHECK_ERROR;

  // Cleanup
  cudaFreeArray ((cudaArray*)array_xdvf);
  cudaFreeArray ((cudaArray*)array_ydvf);
  cudaFreeArray ((cudaArray*)array_zdvf);
  CUDA_CHECK_ERROR;
  cudaFreeArray ((cudaArray*)array_img);
  CUDA_CHECK_ERROR;
  cudaFree (dev_matrix);
  CUDA_CHECK_ERROR;
  cudaFree (dev_IndexInputToPPInput);
  cudaFree (dev_IndexInputToIndexDVF);
  cudaFree (dev_PPInputToIndexInput);
  CUDA_CHECK_ERROR;
}
