/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// rtk includes
#include "rtkCudaLastDimensionTVDenoisingImageFilter.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>

// TEXTURES AND CONSTANTS //

__constant__ SizeValueType4 c_Size;

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__global__ void
denoise_oneD_TV_kernel(float * in, float * out, float beta, float gamma, int niter)
{
  // Allocate a few shared buffers
  extern __shared__ float shared[];
  float *                 input = &shared[0];
  float *                 interm = &shared[c_Size.w];
  float *                 output = &shared[2 * c_Size.w];
  float *                 gradient = &shared[3 * c_Size.w];

  // Each thread reads one element into the shared buffer
  const itk::SizeValueType gindex =
    ((threadIdx.x * c_Size.z + blockIdx.z) * c_Size.y + blockIdx.y) * c_Size.x + blockIdx.x;
  input[threadIdx.x] = in[gindex];
  __syncthreads();

  ///////////////////////////////////////////////////
  // Perform complete 1D TV denoising on the buffer
  // with circular padding border conditions

  // Multiply by beta
  interm[threadIdx.x] = beta * input[threadIdx.x];

  // Compute gradient
  __syncthreads();
  if (threadIdx.x == (c_Size.w - 1))
    gradient[threadIdx.x] = interm[0] - interm[threadIdx.x];
  else
    gradient[threadIdx.x] = interm[threadIdx.x + 1] - interm[threadIdx.x];

  // Magnitude threshold (in 1D, hard threshold on absolute value)
  if (gradient[threadIdx.x] >= 0)
    gradient[threadIdx.x] = fminf(gradient[threadIdx.x], gamma);
  else
    gradient[threadIdx.x] = fmaxf(gradient[threadIdx.x], -gamma);

  // Rest of the iterations
  for (int iter = 0; iter < niter; iter++)
  {
    // Compute divergence
    __syncthreads();
    if (threadIdx.x == 0)
      interm[threadIdx.x] = gradient[threadIdx.x] - gradient[c_Size.w - 1];
    else
      interm[threadIdx.x] = gradient[threadIdx.x] - gradient[threadIdx.x - 1];

    // Subtract and store in output buffer
    __syncthreads();
    output[threadIdx.x] = input[threadIdx.x] - interm[threadIdx.x];

    // Multiply by beta
    __syncthreads();
    interm[threadIdx.x] = output[threadIdx.x] * beta;

    // Compute gradient
    __syncthreads();
    if (threadIdx.x == (c_Size.w - 1))
      gradient[threadIdx.x] -= (interm[0] - interm[threadIdx.x]);
    else
      gradient[threadIdx.x] -= (interm[threadIdx.x + 1] - interm[threadIdx.x]);

    // Magnitude threshold
    __syncthreads();
    if (gradient[threadIdx.x] >= 0)
      gradient[threadIdx.x] = fminf(gradient[threadIdx.x], gamma);
    else
      gradient[threadIdx.x] = fmaxf(gradient[threadIdx.x], -gamma);
  }
  // Done computing 1D TV for this buffer
  ////////////////////////////////////////////////////////

  // Each thread writes one element from the shared buffer into the global memory
  out[gindex] = output[threadIdx.x];
  //  out[gindex] = gindex;
}


//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

void
CUDA_total_variation_last_dimension(int     size[4],
                                    float * dev_in,
                                    float * dev_out,
                                    float   gamma,
                                    float   beta,
                                    int     NumberOfIterations)
{
  SizeValueType4 dev_Size = make_ulonglong4(size[0], size[1], size[2], size[3]);
  cudaMemcpyToSymbol(c_Size, &dev_Size, sizeof(SizeValueType4));

  // Thread Block Dimensions
  // Create one thread block per voxel of the 3D volume.
  // Each thread block handles a single 1D temporal vector
  dim3 dimBlock = dim3(size[3], 1, 1);

  int blocksInX = size[0];
  int blocksInY = size[1];
  int blocksInZ = size[2];

  dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);

  // Dynamic allocation of shared memory
  denoise_oneD_TV_kernel<<<dimGrid, dimBlock, 4 * sizeof(float) * size[3]>>>(
    dev_in, dev_out, beta, gamma, NumberOfIterations);
  CUDA_CHECK_ERROR;
}
