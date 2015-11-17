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
#include "rtkCudaForwardWarpImageFilter.hcu"

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
#include <device_functions.h>

// T E X T U R E S ////////////////////////////////////////////////////////
texture<float, 1, cudaReadModeElementType> tex_IndexInputToPPInputMatrix;
texture<float, 1, cudaReadModeElementType> tex_IndexInputToIndexDVFMatrix;
texture<float, 1, cudaReadModeElementType> tex_PPOutputToIndexOutputMatrix;

texture<float, 3, cudaReadModeElementType> tex_xdvf;
texture<float, 3, cudaReadModeElementType> tex_ydvf;
texture<float, 3, cudaReadModeElementType> tex_zdvf;
///////////////////////////////////////////////////////////////////////////

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__global__
void fillHoles_3Dgrid(float * dev_vol_out, float * dev_accumulate_weights, int3 out_dim)
{
  int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= out_dim.x || j >= out_dim.y || k >= out_dim.z)
    {
    return;
    }

  // Index row major into the volume
  long int out_idx = i + (j + k*out_dim.y)*(out_dim.x);
  long int current_idx;
  int radius = 3;

  float eps = 1e-6;

  // If there is a hole in splat at this point
  if (abs(dev_accumulate_weights[out_idx]) < eps)
    {
    // Replace it with the weighted mean of the neighbours
    // (with dev_accumulate_weights as weights, so as to
    // negate the contribution of neighbouring holes)
    float sum = 0;
    float sum_weights = 0;

    for (int delta_i =-radius; delta_i <= radius; delta_i++)
      {
      for (int delta_j =-radius; delta_j <= radius; delta_j++)
        {
        for (int delta_k =-radius; delta_k <= radius; delta_k++)
          {
          if (   (i + delta_i >= 0) && (i + delta_i < out_dim.x)
              && (j + delta_j >= 0) && (j + delta_j < out_dim.y)
              && (k + delta_k >= 0) && (k + delta_k < out_dim.z))
            {
            current_idx = i + delta_i + (j + delta_j + (k + delta_k)*out_dim.y)*(out_dim.x);
            sum += dev_vol_out[current_idx] * dev_accumulate_weights[current_idx];
            sum_weights += dev_accumulate_weights[current_idx];
            }
          }
        }
      }
    if (abs(sum_weights) > eps)
      dev_vol_out[out_idx] = sum / sum_weights;
    else
      dev_vol_out[out_idx] = 0;
    }
}

__global__
void normalize_3Dgrid(float * dev_vol_out, float * dev_accumulate_weights, int3 out_dim)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= out_dim.x || j >= out_dim.y || k >= out_dim.z)
    {
    return;
    }

  // Index row major into the volume
  long int out_idx = i + (j + k*out_dim.y)*(out_dim.x);

  float eps = 1e-6;

  if (abs(dev_accumulate_weights[out_idx]) > eps)
    dev_vol_out[out_idx] /= dev_accumulate_weights[out_idx];
  else
    dev_vol_out[out_idx] = 0;
}

__global__
void linearSplat_3Dgrid(float * dev_vol_in, float * dev_vol_out, float * dev_accumulate_weights, int3 in_dim, int3 out_dim)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= in_dim.x || j >= in_dim.y || k >= in_dim.z)
    {
    return;
    }

  // Index row major into the volume
  long int in_idx = i + (j + k*in_dim.y)*(in_dim.x);

  // Matrix multiply to get the index in the DVF texture of the current point in the output volume
  float3 IndexInDVF;
  IndexInDVF.x = tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 0)*i + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 1)*j +
         tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 2)*k + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 3);
  IndexInDVF.y = tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 4)*i + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 5)*j +
         tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 6)*k + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 7);
  IndexInDVF.z = tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 8)*i + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 9)*j +
         tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 10)*k + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 11);

  // Get each component of the displacement vector by
  // interpolation in the dvf
  float3 Displacement;
  Displacement.x = tex3D(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.y = tex3D(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.z = tex3D(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

  // Matrix multiply to get the physical coordinates of the current point in the input volume
  float3 PPinInput;
  PPinInput.x = tex1Dfetch(tex_IndexInputToPPInputMatrix, 0)*i + tex1Dfetch(tex_IndexInputToPPInputMatrix, 1)*j +
               tex1Dfetch(tex_IndexInputToPPInputMatrix, 2)*k + tex1Dfetch(tex_IndexInputToPPInputMatrix, 3);
  PPinInput.y = tex1Dfetch(tex_IndexInputToPPInputMatrix, 4)*i + tex1Dfetch(tex_IndexInputToPPInputMatrix, 5)*j +
               tex1Dfetch(tex_IndexInputToPPInputMatrix, 6)*k + tex1Dfetch(tex_IndexInputToPPInputMatrix, 7);
  PPinInput.z = tex1Dfetch(tex_IndexInputToPPInputMatrix, 8)*i + tex1Dfetch(tex_IndexInputToPPInputMatrix, 9)*j +
               tex1Dfetch(tex_IndexInputToPPInputMatrix, 10)*k + tex1Dfetch(tex_IndexInputToPPInputMatrix, 11);

  // Get the index corresponding to the current physical point in output displaced by the displacement vector
  float3 PPDisplaced;
  PPDisplaced.x = PPinInput.x + Displacement.x;
  PPDisplaced.y = PPinInput.y + Displacement.y;
  PPDisplaced.z = PPinInput.z + Displacement.z;

  float3 IndexInOutput;
  IndexInOutput.x =  tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 0) * PPDisplaced.x
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 1) * PPDisplaced.y
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 2) * PPDisplaced.z
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 3);
  IndexInOutput.y =  tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 4) * PPDisplaced.x
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 5) * PPDisplaced.y
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 6) * PPDisplaced.z
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 7);
  IndexInOutput.z =  tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 8) * PPDisplaced.x
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 9) * PPDisplaced.y
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 10)* PPDisplaced.z
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 11);

  // Compute the splat weights
  int3 BaseIndexInOutput;
  BaseIndexInOutput.x = floor(IndexInOutput.x);
  BaseIndexInOutput.y = floor(IndexInOutput.y);
  BaseIndexInOutput.z = floor(IndexInOutput.z);

  float3 Distance;
  Distance.x = IndexInOutput.x - BaseIndexInOutput.x;
  Distance.y = IndexInOutput.y - BaseIndexInOutput.y;
  Distance.z = IndexInOutput.z - BaseIndexInOutput.z;

  float weight000 = (1 - Distance.x) * (1 - Distance.y) * (1 - Distance.z);
  float weight001 = (1 - Distance.x) * (1 - Distance.y) * Distance.z;
  float weight010 = (1 - Distance.x) * Distance.y       * (1 - Distance.z);
  float weight011 = (1 - Distance.x) * Distance.y       * Distance.z;
  float weight100 = Distance.x       * (1 - Distance.y) * (1 - Distance.z);
  float weight101 = Distance.x       * (1 - Distance.y) * Distance.z;
  float weight110 = Distance.x       * Distance.y       * (1 - Distance.z);
  float weight111 = Distance.x       * Distance.y       * Distance.z;

  // Compute indices in the volume
  long int out_idx000 = (BaseIndexInOutput.x + 0) + (BaseIndexInOutput.y + 0) * out_dim.x + (BaseIndexInOutput.z + 0) * out_dim.x * out_dim.y;
  long int out_idx001 = (BaseIndexInOutput.x + 0) + (BaseIndexInOutput.y + 0) * out_dim.x + (BaseIndexInOutput.z + 1) * out_dim.x * out_dim.y;
  long int out_idx010 = (BaseIndexInOutput.x + 0) + (BaseIndexInOutput.y + 1) * out_dim.x + (BaseIndexInOutput.z + 0) * out_dim.x * out_dim.y;
  long int out_idx011 = (BaseIndexInOutput.x + 0) + (BaseIndexInOutput.y + 1) * out_dim.x + (BaseIndexInOutput.z + 1) * out_dim.x * out_dim.y;
  long int out_idx100 = (BaseIndexInOutput.x + 1) + (BaseIndexInOutput.y + 0) * out_dim.x + (BaseIndexInOutput.z + 0) * out_dim.x * out_dim.y;
  long int out_idx101 = (BaseIndexInOutput.x + 1) + (BaseIndexInOutput.y + 0) * out_dim.x + (BaseIndexInOutput.z + 1) * out_dim.x * out_dim.y;
  long int out_idx110 = (BaseIndexInOutput.x + 1) + (BaseIndexInOutput.y + 1) * out_dim.x + (BaseIndexInOutput.z + 0) * out_dim.x * out_dim.y;
  long int out_idx111 = (BaseIndexInOutput.x + 1) + (BaseIndexInOutput.y + 1) * out_dim.x + (BaseIndexInOutput.z + 1) * out_dim.x * out_dim.y;

  // Determine whether they are indeed in the volume
  bool isInVolume_out_idx000 = (BaseIndexInOutput.x + 0 >= 0) && (BaseIndexInOutput.x + 0 < out_dim.x)
                            && (BaseIndexInOutput.y + 0 >= 0) && (BaseIndexInOutput.y + 0 < out_dim.y)
                            && (BaseIndexInOutput.z + 0 >= 0) && (BaseIndexInOutput.z + 0 < out_dim.z);

  bool isInVolume_out_idx001 = (BaseIndexInOutput.x + 0 >= 0) && (BaseIndexInOutput.x + 0 < out_dim.x)
                            && (BaseIndexInOutput.y + 0 >= 0) && (BaseIndexInOutput.y + 0 < out_dim.y)
                            && (BaseIndexInOutput.z + 1 >= 0) && (BaseIndexInOutput.z + 1 < out_dim.z);

  bool isInVolume_out_idx010 = (BaseIndexInOutput.x + 0 >= 0) && (BaseIndexInOutput.x + 0 < out_dim.x)
                            && (BaseIndexInOutput.y + 1 >= 0) && (BaseIndexInOutput.y + 1 < out_dim.y)
                            && (BaseIndexInOutput.z + 0 >= 0) && (BaseIndexInOutput.z + 0 < out_dim.z);

  bool isInVolume_out_idx011 = (BaseIndexInOutput.x + 0 >= 0) && (BaseIndexInOutput.x + 0 < out_dim.x)
                            && (BaseIndexInOutput.y + 1 >= 0) && (BaseIndexInOutput.y + 1 < out_dim.y)
                            && (BaseIndexInOutput.z + 1 >= 0) && (BaseIndexInOutput.z + 1 < out_dim.z);

  bool isInVolume_out_idx100 = (BaseIndexInOutput.x + 1 >= 0) && (BaseIndexInOutput.x + 1 < out_dim.x)
                            && (BaseIndexInOutput.y + 0 >= 0) && (BaseIndexInOutput.y + 0 < out_dim.y)
                            && (BaseIndexInOutput.z + 0 >= 0) && (BaseIndexInOutput.z + 0 < out_dim.z);

  bool isInVolume_out_idx101 = (BaseIndexInOutput.x + 1 >= 0) && (BaseIndexInOutput.x + 1 < out_dim.x)
                            && (BaseIndexInOutput.y + 0 >= 0) && (BaseIndexInOutput.y + 0 < out_dim.y)
                            && (BaseIndexInOutput.z + 1 >= 0) && (BaseIndexInOutput.z + 1 < out_dim.z);

  bool isInVolume_out_idx110 = (BaseIndexInOutput.x + 1 >= 0) && (BaseIndexInOutput.x + 1 < out_dim.x)
                            && (BaseIndexInOutput.y + 1 >= 0) && (BaseIndexInOutput.y + 1 < out_dim.y)
                            && (BaseIndexInOutput.z + 0 >= 0) && (BaseIndexInOutput.z + 0 < out_dim.z);

  bool isInVolume_out_idx111 = (BaseIndexInOutput.x + 1 >= 0) && (BaseIndexInOutput.x + 1 < out_dim.x)
                            && (BaseIndexInOutput.y + 1 >= 0) && (BaseIndexInOutput.y + 1 < out_dim.y)
                            && (BaseIndexInOutput.z + 1 >= 0) && (BaseIndexInOutput.z + 1 < out_dim.z);

  // Perform splat if voxel is indeed in output volume
  if (isInVolume_out_idx000)
    {
    atomicAdd(&dev_vol_out[out_idx000], dev_vol_in[in_idx] * weight000);
    atomicAdd(&dev_accumulate_weights[out_idx000], weight000);
    }
  if (isInVolume_out_idx001)
    {
    atomicAdd(&dev_vol_out[out_idx001], dev_vol_in[in_idx] * weight001);
    atomicAdd(&dev_accumulate_weights[out_idx001], weight001);
    }
  if (isInVolume_out_idx010)
    {
    atomicAdd(&dev_vol_out[out_idx010], dev_vol_in[in_idx] * weight010);
    atomicAdd(&dev_accumulate_weights[out_idx010], weight010);
    }
  if (isInVolume_out_idx011)
    {
    atomicAdd(&dev_vol_out[out_idx011], dev_vol_in[in_idx] * weight011);
    atomicAdd(&dev_accumulate_weights[out_idx011], weight011);
    }
  if (isInVolume_out_idx100)
    {
    atomicAdd(&dev_vol_out[out_idx100], dev_vol_in[in_idx] * weight100);
    atomicAdd(&dev_accumulate_weights[out_idx100], weight100);
    }
  if (isInVolume_out_idx101)
    {
    atomicAdd(&dev_vol_out[out_idx101], dev_vol_in[in_idx] * weight101);
    atomicAdd(&dev_accumulate_weights[out_idx101], weight101);
    }
  if (isInVolume_out_idx110)
    {
    atomicAdd(&dev_vol_out[out_idx110], dev_vol_in[in_idx] * weight110);
    atomicAdd(&dev_accumulate_weights[out_idx110], weight110);
    }
  if (isInVolume_out_idx111)
    {
    atomicAdd(&dev_vol_out[out_idx111], dev_vol_in[in_idx] * weight111);
    atomicAdd(&dev_accumulate_weights[out_idx111], weight111);
    }
}

__global__
void nearestNeighborSplat_3Dgrid(float * dev_vol_in, float * dev_vol_out, float * dev_accumulate_weights, int3 in_dim, int3 out_dim)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= in_dim.x || j >= in_dim.y || k >= in_dim.z)
    {
    return;
    }

  // Index row major into the volume
  long int in_idx = i + (j + k*in_dim.y)*(in_dim.x);

  // Matrix multiply to get the index in the DVF texture of the current point in the output volume
  float3 IndexInDVF;
  IndexInDVF.x = tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 0)*i + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 1)*j +
         tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 2)*k + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 3);
  IndexInDVF.y = tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 4)*i + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 5)*j +
         tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 6)*k + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 7);
  IndexInDVF.z = tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 8)*i + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 9)*j +
         tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 10)*k + tex1Dfetch(tex_IndexInputToIndexDVFMatrix, 11);

  // Get each component of the displacement vector by
  // interpolation in the dvf
  float3 Displacement;
  Displacement.x = tex3D(tex_xdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.y = tex3D(tex_ydvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);
  Displacement.z = tex3D(tex_zdvf, IndexInDVF.x + 0.5f, IndexInDVF.y + 0.5f, IndexInDVF.z + 0.5f);

  // Matrix multiply to get the physical coordinates of the current point in the input volume
  float3 PPinInput;
  PPinInput.x = tex1Dfetch(tex_IndexInputToPPInputMatrix, 0)*i + tex1Dfetch(tex_IndexInputToPPInputMatrix, 1)*j +
               tex1Dfetch(tex_IndexInputToPPInputMatrix, 2)*k + tex1Dfetch(tex_IndexInputToPPInputMatrix, 3);
  PPinInput.y = tex1Dfetch(tex_IndexInputToPPInputMatrix, 4)*i + tex1Dfetch(tex_IndexInputToPPInputMatrix, 5)*j +
               tex1Dfetch(tex_IndexInputToPPInputMatrix, 6)*k + tex1Dfetch(tex_IndexInputToPPInputMatrix, 7);
  PPinInput.z = tex1Dfetch(tex_IndexInputToPPInputMatrix, 8)*i + tex1Dfetch(tex_IndexInputToPPInputMatrix, 9)*j +
               tex1Dfetch(tex_IndexInputToPPInputMatrix, 10)*k + tex1Dfetch(tex_IndexInputToPPInputMatrix, 11);

  // Get the index corresponding to the current physical point in output displaced by the displacement vector
  float3 PPDisplaced;
  PPDisplaced.x = PPinInput.x + Displacement.x;
  PPDisplaced.y = PPinInput.y + Displacement.y;
  PPDisplaced.z = PPinInput.z + Displacement.z;

  float3 IndexInOutput;
  IndexInOutput.x =  floor(tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 0) * PPDisplaced.x
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 1) * PPDisplaced.y
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 2) * PPDisplaced.z
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 3) + 0.5);
  IndexInOutput.y =  floor(tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 4) * PPDisplaced.x
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 5) * PPDisplaced.y
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 6) * PPDisplaced.z
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 7) + 0.5);
  IndexInOutput.z =  floor(tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 8) * PPDisplaced.x
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 9) * PPDisplaced.y
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 10)* PPDisplaced.z
                  + tex1Dfetch(tex_PPOutputToIndexOutputMatrix, 11) + 0.5);

  bool isInVolume = (IndexInOutput.x >= 0) && (IndexInOutput.x < out_dim.x)
                 && (IndexInOutput.y >= 0) && (IndexInOutput.y < out_dim.y)
                 && (IndexInOutput.z >= 0) && (IndexInOutput.z < out_dim.z);

  // Perform splat if voxel is indeed in output volume
  if (isInVolume)
    {
    long int out_idx = IndexInOutput.x + IndexInOutput.y * out_dim.x + IndexInOutput.z * out_dim.x * out_dim.y;
    atomicAdd(&dev_vol_out[out_idx], dev_vol_in[in_idx]);
    atomicAdd(&dev_accumulate_weights[out_idx], 1);
    }
}


//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_ForwardWarp /////////////////////////////
void
CUDA_ForwardWarp(int input_vol_dim[3],
    int input_dvf_dim[3],
    int output_vol_dim[3],
    float IndexInputToPPInputMatrix[12],
    float IndexInputToIndexDVFMatrix[12],
    float PPOutputToIndexOutputMatrix[12],
    float *dev_input_vol,
    float *dev_input_xdvf,
    float *dev_input_ydvf,
    float *dev_input_zdvf,
    float *dev_output_vol,
    bool isLinear)
{

  // Prepare channel description for arrays
  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

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

  // Allocate the arrays
  cudaArray *array_xdvf;
  cudaArray *array_ydvf;
  cudaArray *array_zdvf;
  cudaMalloc3DArray((cudaArray**)&array_xdvf, &channelDesc, dvfExtent);
  cudaMalloc3DArray((cudaArray**)&array_ydvf, &channelDesc, dvfExtent);
  cudaMalloc3DArray((cudaArray**)&array_zdvf, &channelDesc, dvfExtent);
  CUDA_CHECK_ERROR;

  // Copy image data to arrays. For a nice explanation on make_cudaPitchedPtr, checkout
  // http://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
  cudaMemcpy3DParms xCopyParams = {0};
  xCopyParams.srcPtr   = make_cudaPitchedPtr(dev_input_xdvf, input_dvf_dim[0] * sizeof(float), input_dvf_dim[0], input_dvf_dim[1]);
  xCopyParams.dstArray = (cudaArray*)array_xdvf;
  xCopyParams.extent   = dvfExtent;
  xCopyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&xCopyParams);
  CUDA_CHECK_ERROR;

  cudaMemcpy3DParms yCopyParams = {0};
  yCopyParams.srcPtr   = make_cudaPitchedPtr(dev_input_ydvf, input_dvf_dim[0] * sizeof(float), input_dvf_dim[0], input_dvf_dim[1]);
  yCopyParams.dstArray = (cudaArray*)array_ydvf;
  yCopyParams.extent   = dvfExtent;
  yCopyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&yCopyParams);
  CUDA_CHECK_ERROR;

  cudaMemcpy3DParms zCopyParams = {0};
  zCopyParams.srcPtr   = make_cudaPitchedPtr(dev_input_zdvf, input_dvf_dim[0] * sizeof(float), input_dvf_dim[0], input_dvf_dim[1]);
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

  float *dev_PPOutputToIndexOutput;
  cudaMalloc( (void**)&dev_PPOutputToIndexOutput, 12*sizeof(float) );
  cudaMemcpy (dev_PPOutputToIndexOutput, PPOutputToIndexOutputMatrix, 12*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTexture (0, tex_PPOutputToIndexOutputMatrix, dev_PPOutputToIndexOutput, 12*sizeof(float) );

  ///////////////////////////////////////
  /// Initialize the output

  cudaMemset((void *)dev_output_vol, 0, sizeof(float) * output_vol_dim[0] * output_vol_dim[1] * output_vol_dim[2]);

  //////////////////////////////////////
  /// Create an image to store the splat weights
  /// (in order to normalize)

  float *dev_accumulate_weights;
  cudaMalloc( (void**)&dev_accumulate_weights, sizeof(float) * output_vol_dim[0] * output_vol_dim[1] * output_vol_dim[2]);
  cudaMemset((void *)dev_accumulate_weights, 0, sizeof(float) * output_vol_dim[0] * output_vol_dim[1] * output_vol_dim[2]);

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

  dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

  // Note: the DVF is passed via texture memory
  //-------------------------------------
  if (isLinear)
    linearSplat_3Dgrid <<< dimGrid, dimBlock >>> ( dev_input_vol,
                                            dev_output_vol,
                                            dev_accumulate_weights,
                                            make_int3(input_vol_dim[0], input_vol_dim[1], input_vol_dim[2]),
                                            make_int3(output_vol_dim[0], output_vol_dim[1], output_vol_dim[2]));
  else
    nearestNeighborSplat_3Dgrid <<< dimGrid, dimBlock >>> ( dev_input_vol,
                                              dev_output_vol,
                                              dev_accumulate_weights,
                                              make_int3(input_vol_dim[0], input_vol_dim[1], input_vol_dim[2]),
                                              make_int3(output_vol_dim[0], output_vol_dim[1], output_vol_dim[2]));

  CUDA_CHECK_ERROR;

  normalize_3Dgrid <<< dimGrid, dimBlock >>> ( dev_output_vol,
                                               dev_accumulate_weights,
                                               make_int3(output_vol_dim[0], output_vol_dim[1], output_vol_dim[2]));
  CUDA_CHECK_ERROR;

  fillHoles_3Dgrid <<< dimGrid, dimBlock >>> ( dev_output_vol,
                                               dev_accumulate_weights,
                                               make_int3(output_vol_dim[0], output_vol_dim[1], output_vol_dim[2]));
  CUDA_CHECK_ERROR;

  // Unbind the image and projection matrix textures
  cudaUnbindTexture (tex_xdvf);
  cudaUnbindTexture (tex_ydvf);
  cudaUnbindTexture (tex_zdvf);
  CUDA_CHECK_ERROR;
  cudaUnbindTexture (tex_IndexInputToPPInputMatrix);
  cudaUnbindTexture (tex_PPOutputToIndexOutputMatrix);
  cudaUnbindTexture (tex_IndexInputToIndexDVFMatrix);
  CUDA_CHECK_ERROR;

  // Cleanup
  cudaFreeArray ((cudaArray*)array_xdvf);
  cudaFreeArray ((cudaArray*)array_ydvf);
  cudaFreeArray ((cudaArray*)array_zdvf);
  CUDA_CHECK_ERROR;
  cudaFree (dev_accumulate_weights);
  CUDA_CHECK_ERROR;
  cudaFree (dev_IndexInputToPPInput);
  cudaFree (dev_PPOutputToIndexOutput);
  cudaFree (dev_IndexInputToIndexDVF);
  CUDA_CHECK_ERROR;
}
