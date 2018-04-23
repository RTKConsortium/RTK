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

// rtk includes
#include "rtkCudaConstantVolumeSeriesSource.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>

// TEXTURES AND CONSTANTS //

__constant__ int4 c_Size;

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_


__global__
void
set_volume_series_to_constant(float *out, float value)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= c_Size.x || j >= c_Size.y || k >= c_Size.z * c_Size.w)
      return;

  long int id = (k * c_Size.y + j) * c_Size.x + i;

  out[id] = value;
}


//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

void
CUDA_generate_constant_volume_series(int size[4],
                                     float* dev_out,
                                     float constantValue)
{
  int4 dev_Size = make_int4(size[0], size[1], size[2], size[3]);
  cudaMemcpyToSymbol(c_Size, &dev_Size, sizeof(int4));

  // NOTE : memset sets every BYTE of the memory to the value given in
  // argument (here, 0). With 0, it's fine, as all number formats seem
  // to represent 0 by bytes containing only zeros.
  // But running memset with argument 1, for example, will set all bytes
  // of the memory chunk to 00000001. And so reading a float from this memory
  // chunk means reading "00000001 00000001 00000001 00000001" and
  // interpreting it as a float, which certainly doesn't give 1.0
  // Therefore we use memset to set the image to zero quickly (most of the
  // time, that's what rtkConstantImageSource is used for) and if necessary,
  // run a kernel to replace the zeros with constantValue.

  // Reset output volume
  long int memorySizeOutput = size[0] * size[1] * size[2] * size[3] * sizeof(float);
  cudaMemset((void *)dev_out, 0, memorySizeOutput );

  if (!(constantValue == 0))
    {
    // Thread Block Dimensions
    dim3 dimBlock = dim3(4, 4, 16);

    int blocksInX = iDivUp(size[0], dimBlock.x);
    int blocksInY = iDivUp(size[1], dimBlock.y);
    int blocksInZ = iDivUp(size[2] * size[3], dimBlock.z);

    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);

    set_volume_series_to_constant <<< dimGrid, dimBlock >>> ( dev_out, constantValue);
    }

  CUDA_CHECK_ERROR;
}
