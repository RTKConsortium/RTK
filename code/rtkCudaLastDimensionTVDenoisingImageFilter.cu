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
#include "rtkCudaLastDimensionTVDenoisingImageFilter.hcu"
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
denoise_oneD_TV_kernel(float *in, float* out, float beta, float gamma, int niter)
{
  // Allocate a few shared buffers
  extern __shared__ float shared[];
  float* input = &shared[0];
  float* interm = &shared[c_Size.w];
  float* output = &shared[2 * c_Size.w];
  float* gradient = &shared[3 * c_Size.w];

  // Each thread reads one element into the shared buffer
  long int gindex = ((threadIdx.x * c_Size.z + blockIdx.z) * c_Size.y + blockIdx.y) * c_Size.x + blockIdx.x;
  int lindex = threadIdx.x;
  input[lindex] = in[gindex];
  __syncthreads();

  ///////////////////////////////////////////////////
  // Perform complete 1D TV denoising on the buffer
  // with circular padding border conditions

  // Multiply by beta
  interm[lindex] = beta * input[lindex];

  // Compute gradient
  __syncthreads();
  if (lindex == (c_Size.w - 1))
    gradient[lindex] = interm[0] - interm[lindex];
  else
    gradient[lindex] = interm[lindex + 1] - interm[lindex];

  // Magnitude threshold (in 1D, hard threshold on absolute value)
  if (gradient[lindex] >= 0)
    gradient[lindex] = fminf(gradient[lindex], gamma);
  else
    gradient[lindex] = fmaxf(gradient[lindex], -gamma);

  // Rest of the iterations
  for (int iter=0; iter<niter; iter++)
    {
      // Compute divergence
      __syncthreads();
      if (lindex == 0)
        interm[lindex] = gradient[lindex] - gradient[c_Size.w - 1];
      else
        interm[lindex] = gradient[lindex] - gradient[lindex - 1];

      // Subtract and store in output buffer
      __syncthreads();
      output[lindex] = input[lindex] - interm[lindex];

      // Multiply by beta
      __syncthreads();
      interm[lindex] = output[lindex] * beta;

      // Compute gradient
      __syncthreads();
      if (lindex == (c_Size.w - 1))
        gradient[lindex] -= (interm[0] - interm[lindex]);
      else
        gradient[lindex] -= (interm[lindex + 1] - interm[lindex]);

      // Magnitude threshold
      __syncthreads();
      if (gradient[lindex] >= 0)
        gradient[lindex] = fminf(gradient[lindex], gamma);
      else
        gradient[lindex] = fmaxf(gradient[lindex], -gamma);
    }
  // Done computing 1D TV for this buffer
  ////////////////////////////////////////////////////////

  // Each thread writes one element from the shared buffer into the global memory
  out[gindex] = output[lindex];
//  out[gindex] = gindex;
}


//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

void
CUDA_total_variation_last_dimension(int size[4],
                                   float* dev_in,
                                   float* dev_out,
                                   float gamma,
                                   float beta,
                                   int NumberOfIterations)
{
  int4 dev_Size = make_int4(size[0], size[1], size[2], size[3]);
  cudaMemcpyToSymbol(c_Size, &dev_Size, sizeof(int4));

  // Thread Block Dimensions
  // Create one thread block per voxel of the 3D volume.
  // Each thread block handles a single 1D temporal vector
  dim3 dimBlock = dim3(size[3], 1, 1);

  int blocksInX = size[0];
  int blocksInY = size[1];
  int blocksInZ = size[2];

  dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);

  // Dynamic allocation of shared memory
  denoise_oneD_TV_kernel <<< dimGrid, dimBlock, 4*sizeof(float)*size[3] >>> (dev_in, dev_out, beta, gamma, NumberOfIterations);
  CUDA_CHECK_ERROR;
}
