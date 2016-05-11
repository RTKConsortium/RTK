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
#include "rtkCudaWarpForwardProjectionImageFilter.hcu"

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

// TEXTURES AND CONSTANTS //
texture<float, 3, cudaReadModeElementType> tex_xdvf;
texture<float, 3, cudaReadModeElementType> tex_ydvf;
texture<float, 3, cudaReadModeElementType> tex_zdvf;
texture<float, 3, cudaReadModeElementType> tex_vol;

__constant__ float3 c_sourcePos;
__constant__ int2 c_projSize;
__constant__ float3 c_boxMin;
__constant__ float3 c_boxMax;
__constant__ float3 c_spacing;
__constant__ int3 c_volSize;
__constant__ float c_tStep;

__constant__ float c_matrix[12];
__constant__ float c_IndexInputToPPInputMatrix[12];
__constant__ float c_IndexInputToIndexDVFMatrix[12];
__constant__ float c_PPInputToIndexInputMatrix[12];

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

// KERNEL kernel_forwardProject
__global__
void kernel_warped_forwardProject(float *dev_proj_in, float *dev_proj_out)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int numThread = j*c_projSize.x + i;

  if (i >= c_projSize.x || j >= c_projSize.y)
    return;

  // Setting ray origin
  Ray ray;
  ray.o = c_sourcePos;

  float3 pixelPos = matrix_multiply(make_float3(i,j,0), c_matrix);

  ray.d = pixelPos - ray.o;
  ray.d = ray.d / sqrtf(dot(ray.d,ray.d));

  // Detect intersection with box
  float tnear, tfar;
  if ( !intersectBox(ray, &tnear, &tfar, c_boxMin, c_boxMax) || tfar < 0.f )
    {
    dev_proj_out[numThread] = dev_proj_in[numThread];
    }
  else
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
    float  sample = 0.0f;
    float  sum    = 0.0f;
    for(t=tnear; t<=tfar; t+=vStep)
      {
      float3 IndexInDVF = matrix_multiply(pos, c_IndexInputToIndexDVFMatrix);

      // Get each component of the displacement vector by
      // interpolation in the dvf
      float3 Displacement;
      Displacement.x = tex3D(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
      Displacement.y = tex3D(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
      Displacement.z = tex3D(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

      // Matrix multiply to get the physical coordinates of the current point in the output volume
      float3 PP = matrix_multiply(pos, c_IndexInputToPPInputMatrix);
      
      // Get the index corresponding to the current physical point in output displaced by the displacement vector
      PP += Displacement;

      // Convert it to a continuous index
      float3 IndexInInput = matrix_multiply(PP, c_PPInputToIndexInputMatrix);

      // Read from 3D texture from volume
      sample = tex3D(tex_vol, IndexInInput.x, IndexInInput.y, IndexInInput.z);

      // Accumulate, and move forward along the ray
      sum += sample;
      pos += step;
      }
    dev_proj_out[numThread] = dev_proj_in[numThread] + (sum+(tfar-t+halfVStep)/vStep*sample) * c_tStep;
    }
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_forward_project() //////////////////////////////////
void
CUDA_warp_forward_project( int projections_size[2],
                      int vol_size[3],
                      int dvf_size[3],
                      float matrix[12],
                      float *dev_proj_in,
                      float *dev_proj_out,
                      float *dev_vol,
                      float t_step,
                      double source_position[3],
                      float box_min[3],
                      float box_max[3],
                      float spacing[3],
		      float *dev_input_dvf,
		      float IndexInputToIndexDVFMatrix[12],
		      float PPInputToIndexInputMatrix[12],
		      float IndexInputToPPInputMatrix[12]
		    )
{
  // Create CUBLAS context
  cublasHandle_t  handle;
  cublasCreate(&handle);

  // Prepare channel description for arrays
  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  
  // Extent stuff, will be used for each component extraction
  cudaExtent dvfExtent = make_cudaExtent(dvf_size[0], dvf_size[1], dvf_size[2]);

  // Set texture parameters for the input volume
  tex_vol.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
  tex_vol.addressMode[1] = cudaAddressModeClamp;
  tex_vol.addressMode[2] = cudaAddressModeClamp;
  tex_vol.normalized = false;                     // access with normalized texture coordinates
  tex_vol.filterMode = cudaFilterModeLinear;      // linear interpolation

  // Copy volume data to array, bind the array to the texture
  cudaExtent volExtent = make_cudaExtent(vol_size[0], vol_size[1], vol_size[2]);
  cudaArray *array_vol;
  cudaMalloc3DArray((cudaArray**)&array_vol, &channelDesc, volExtent);
  CUDA_CHECK_ERROR;

  // Copy data to 3D array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr   = make_cudaPitchedPtr(dev_vol, vol_size[0]*sizeof(float), vol_size[0], vol_size[1]);
  copyParams.dstArray = (cudaArray*)array_vol;
  copyParams.extent   = volExtent;
  copyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&copyParams);
  CUDA_CHECK_ERROR;

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
  cudaMemcpyToSymbol (c_IndexInputToPPInputMatrix, IndexInputToPPInputMatrix, 12*sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (c_IndexInputToIndexDVFMatrix,  IndexInputToIndexDVFMatrix,  12*sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (c_PPInputToIndexInputMatrix,  PPInputToIndexInputMatrix,  12*sizeof(float), 0, cudaMemcpyHostToDevice);

  // constant memory
  float3 dev_sourcePos = make_float3(source_position[0], source_position[1], source_position[2]);
  float3 dev_boxMin = make_float3(box_min[0], box_min[1], box_min[2]);
  int2 dev_projSize = make_int2(projections_size[0], projections_size[1]);
  float3 dev_boxMax = make_float3(box_max[0], box_max[1], box_max[2]);
  float3 dev_spacing = make_float3(spacing[0], spacing[1], spacing[2]);
  int3 dev_vol_size = make_int3(vol_size[0], vol_size[1], vol_size[2]);
  cudaMemcpyToSymbol(c_sourcePos, &dev_sourcePos, sizeof(float3));
  cudaMemcpyToSymbol(c_projSize, &dev_projSize, sizeof(int2));
  cudaMemcpyToSymbol(c_boxMin, &dev_boxMin, sizeof(float3));
  cudaMemcpyToSymbol(c_boxMax, &dev_boxMax, sizeof(float3));
  cudaMemcpyToSymbol(c_spacing, &dev_spacing, sizeof(float3));
  cudaMemcpyToSymbol(c_tStep, &t_step, sizeof(float));
  cudaMemcpyToSymbol(c_volSize, &dev_vol_size, sizeof(int3));

  ///////////////
  // RUN
  dim3 dimBlock  = dim3(16, 16, 1);
  dim3 dimGrid = dim3(iDivUp(projections_size[0], dimBlock.x), iDivUp(projections_size[1], dimBlock.y));

  // Bind 3D array to 3D texture
  cudaBindTextureToArray(tex_vol, (cudaArray*)array_vol, channelDesc);
  CUDA_CHECK_ERROR;

  kernel_warped_forwardProject <<< dimGrid, dimBlock >>> (dev_proj_in, dev_proj_out);

  cudaUnbindTexture (tex_xdvf);
  cudaUnbindTexture (tex_ydvf);
  cudaUnbindTexture (tex_zdvf);
  cudaUnbindTexture (tex_vol);
  CUDA_CHECK_ERROR;

  cudaFreeArray ((cudaArray*) DVFcomponentArrays[0]);
  cudaFreeArray ((cudaArray*) DVFcomponentArrays[1]);
  cudaFreeArray ((cudaArray*) DVFcomponentArrays[2]);
  delete[] DVFcomponentArrays;
  cudaFreeArray ((cudaArray*)array_vol);
  CUDA_CHECK_ERROR;

  // Destroy CUBLAS context
  cublasDestroy(handle);
}
