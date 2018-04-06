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
#include "rtkCudaTotalVariationDenoisingBPDQImageFilter.hcu"
#include "rtkCudaFirstOrderKernels.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>

// TEXTURES AND CONSTANTS //

__constant__ int3 c_Size;
__constant__ float3 c_Spacing;

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
__global__
void
magnitude_threshold_kernel(float * grad_x, float * grad_y, float * grad_z, float gamma)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= c_Size.x || j >= c_Size.y  || k >= c_Size.z)
      return;

  long int id   = (k     * c_Size.y + j)    * c_Size.x + i;

  float norm = sqrt(grad_x[id] * grad_x[id] + grad_y[id] * grad_y[id] + grad_z[id] * grad_z[id]);
  if (norm > gamma )
    {
    float ratio = gamma / norm;
    grad_x[id] *= ratio;
    grad_y[id] *= ratio;
    grad_z[id] *= ratio;
    }

}

__global__
void
gradient_and_subtract_kernel(float * in, float * grad_x, float * grad_y, float * grad_z)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= c_Size.x || j >= c_Size.y  || k >= c_Size.z)
      return;

  long int id   = (k     * c_Size.y + j)    * c_Size.x + i;
  long int id_x = (k     * c_Size.y + j)    * c_Size.x + i + 1;
  long int id_y = (k     * c_Size.y + j + 1)* c_Size.x + i;
  long int id_z = ((k+1) * c_Size.y + j)    * c_Size.x + i;

  if (i != (c_Size.x - 1)) grad_x[id] -= ((in[id_x] - in[id]) / c_Spacing.x);
  if (j != (c_Size.y - 1)) grad_y[id] -= ((in[id_y] - in[id]) / c_Spacing.y);
  if (k != (c_Size.z - 1)) grad_z[id] -= ((in[id_z] - in[id]) / c_Spacing.z);
}

__global__
void
multiply_by_beta_kernel(float *input, float* output, float beta)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= c_Size.x || j >= c_Size.y || k >= c_Size.z)
      return;

  long int id = (k * c_Size.y + j) * c_Size.x + i;

  output[id] = input[id] * beta;
}

__global__
void
subtract_kernel(float *in1, float* in2, float* out)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= c_Size.x || j >= c_Size.y || k >= c_Size.z)
      return;

  long int id = (k * c_Size.y + j) * c_Size.x + i;

  out[id] = in1[id] - in2[id];
}


//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

void
CUDA_total_variation( int size[3],
                      float spacing[3],
                      float* dev_in,
                      float* dev_out,
                      float gamma,
                      float beta,
                      int NumberOfIterations)
{
  int3 dev_Size = make_int3(size[0], size[1], size[2]);
  cudaMemcpyToSymbol(c_Size, &dev_Size, sizeof(int3));

  float3 dev_Spacing = make_float3(spacing[0], spacing[1], spacing[2]);
  cudaMemcpyToSymbol(c_Spacing, &dev_Spacing, sizeof(float3));

  float minSpacing = spacing[0];
  if (spacing[1] < minSpacing) minSpacing = spacing[1];
  if (spacing[2] < minSpacing) minSpacing = spacing[2];

  // Reset output volume
  long int memorySizeOutput = size[0] * size[1] * size[2] * sizeof(float);
  cudaMemset((void *)dev_out, 0, memorySizeOutput );

  // Initialize volume to store intermediate images
  float * dev_interm;
  cudaMalloc( (void**)&dev_interm, memorySizeOutput);
  cudaMemset(dev_interm, 0, memorySizeOutput);

  // Initialize volumes to store the gradient components
  float * dev_grad_x;
  float * dev_grad_y;
  float * dev_grad_z;
  cudaMalloc( (void**)&dev_grad_x, memorySizeOutput);
  cudaMalloc( (void**)&dev_grad_y, memorySizeOutput);
  cudaMalloc( (void**)&dev_grad_z, memorySizeOutput);
  cudaMemset(dev_grad_x, 0, memorySizeOutput);
  cudaMemset(dev_grad_y, 0, memorySizeOutput);
  cudaMemset(dev_grad_z, 0, memorySizeOutput);

  // Thread Block Dimensions
  dim3 dimBlock = dim3(16, 4, 4);

  int blocksInX = iDivUp(size[0], dimBlock.x);
  int blocksInY = iDivUp(size[1], dimBlock.y);
  int blocksInZ = iDivUp(size[2], dimBlock.z);

  dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);

  // First iteration
  multiply_by_beta_kernel <<< dimGrid, dimBlock >>> ( dev_in, dev_interm, beta);
  CUDA_CHECK_ERROR;

  gradient_kernel <<< dimGrid, dimBlock >>> ( dev_interm, dev_grad_x, dev_grad_y, dev_grad_z, dev_Size, dev_Spacing);
  CUDA_CHECK_ERROR;

  magnitude_threshold_kernel <<< dimGrid, dimBlock >>> ( dev_grad_x, dev_grad_y, dev_grad_z, gamma);
  CUDA_CHECK_ERROR;

  // Rest of the iterations
  for (int iter=0; iter<NumberOfIterations; iter++)
    {
    divergence_kernel <<< dimGrid, dimBlock >>> ( dev_grad_x, dev_grad_y, dev_grad_z, dev_interm, dev_Size, dev_Spacing);
    CUDA_CHECK_ERROR;

    subtract_kernel <<< dimGrid, dimBlock >>> ( dev_in, dev_interm, dev_out);
    CUDA_CHECK_ERROR;

    multiply_by_beta_kernel <<< dimGrid, dimBlock >>> ( dev_out, dev_interm, beta * minSpacing);
    CUDA_CHECK_ERROR;

    gradient_and_subtract_kernel <<< dimGrid, dimBlock >>> ( dev_interm, dev_grad_x, dev_grad_y, dev_grad_z);
    CUDA_CHECK_ERROR;

    magnitude_threshold_kernel <<< dimGrid, dimBlock >>> ( dev_grad_x, dev_grad_y, dev_grad_z, gamma);
    CUDA_CHECK_ERROR;
    }

  // Cleanup
  cudaFree (dev_interm);
  cudaFree (dev_grad_x);
  cudaFree (dev_grad_y);
  cudaFree (dev_grad_z);
  CUDA_CHECK_ERROR;
}
