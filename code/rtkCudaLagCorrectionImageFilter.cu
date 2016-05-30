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

#include "rtkCudaLagCorrectionImageFilter.hcu"
#include "rtkCudaUtilities.hcu"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__constant__ float cst_coef[9];

__global__
void kernel_lag_correction(
int3 proj_idx_in,
int3 proj_size_in,
int2 proj_size_in_buf,
int3 proj_idx_out,
int3 proj_size_out,
int2 proj_size_out_buf,
unsigned short *dev_proj_in,
unsigned short *dev_proj_out,
float *state
)
{
  const int modelOrder = 4;

  // compute thread index
  int3 tIdx;
  tIdx.x = blockIdx.x * blockDim.x + threadIdx.x;
  tIdx.y = blockIdx.y * blockDim.y + threadIdx.y;
  tIdx.z = blockIdx.z * blockDim.z + threadIdx.z;
  long int tIdx_comp = tIdx.x + tIdx.y * proj_size_out.x + tIdx.z * proj_size_out_buf.x * proj_size_out_buf.y;

  // check if outside of projection grid
  if (tIdx.x >= proj_size_out.x || tIdx.y >= proj_size_out.y || tIdx.z >= proj_size_out.z)
    return;

  // compute projection index from thread index
  int3 pIdx = make_int3(tIdx.x + proj_idx_out.x,
    tIdx.y + proj_idx_out.y,
    tIdx.z + proj_idx_out.z);
  // combined proj. index -> use thread index in z because accessing memory only with this index
  long int pIdx_comp = (pIdx.x - proj_idx_in.x) + (pIdx.y - proj_idx_in.y) * proj_size_in_buf.x + (pIdx.z - proj_idx_in.z) * proj_size_in_buf.x * proj_size_in_buf.y;
  
  long int sIdx_comp = tIdx.x + tIdx.y * proj_size_out.x;
  unsigned idx_s = sIdx_comp*modelOrder;
  
  float yk = static_cast<float>(dev_proj_in[pIdx_comp]);
  float xk = yk;
  
  float Sa[modelOrder];
  for (unsigned int n = 0; n < modelOrder; n++)
  {
    // Compute the update of internal state for nth exponential
    float expmA_n = cst_coef[4 + n];
    Sa[n] = expmA_n*state[idx_s + n];

    // Update x[k] by removing contribution of the nth exponential
    float B_n = cst_coef[n];
    xk -= B_n * Sa[n];
  }

  // Apply normalization factor
  xk = xk / cst_coef[8];

  // Update internal state Snk
  for (unsigned int n = 0; n < modelOrder; n++) {
    state[idx_s + n] = xk + Sa[n];
  }

  // Avoid negative values
  xk = (xk < 0.0f) ? 0.f : xk;

  dev_proj_out[tIdx_comp] = static_cast<unsigned short>(xk);
}

void
CUDA_lag_correction(
int proj_idx_in[3], // overlapping input region index
int proj_dim_in[3], // overlapping input region size
int proj_dim_in_buf[2], // input size of buffered region
int proj_idx_out[3], // output region index
int proj_dim_out[3], // output region size
int proj_dim_out_buf[2], // output size of buffered region
unsigned short *dev_proj_in,
unsigned short *dev_proj_out,
float *h_state,
int state_size, float *coefficients)
{
  // Thread Block Dimensions
  int tBlock_x = 16;
  int tBlock_y = 16;
  int tBlock_z = 2;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int  blocksInX = (proj_dim_out[0] - 1) / tBlock_x + 1;
  unsigned int  blocksInY = (proj_dim_out[1] - 1) / tBlock_y + 1;
  unsigned int  blocksInZ = (proj_dim_out[2] - 1) / tBlock_z + 1;

  float *d_state;  // device state
  cudaMalloc((void**)&d_state, state_size);
  if (cudaMemcpy(d_state, h_state, state_size, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cout << "Error allocating state" << std::endl;
  }
  
  if (coefficients[8] <= 0.0)
    coefficients[8] = 1.0;

  cudaMemcpyToSymbol(cst_coef, coefficients, 9 * sizeof(float));

  dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
  kernel_lag_correction <<< dimGrid, dimBlock >>> (
    make_int3(proj_idx_in[0], proj_idx_in[1], proj_idx_in[2]),
    make_int3(proj_dim_in[0], proj_dim_in[1], proj_dim_in[2]),
    make_int2(proj_dim_in_buf[0], proj_dim_in_buf[1]),
    make_int3(proj_idx_out[0], proj_idx_out[1], proj_idx_out[2]),
    make_int3(proj_dim_out[0], proj_dim_out[1], proj_dim_out[2]),
    make_int2(proj_dim_out_buf[0], proj_dim_out_buf[1]),
    dev_proj_in,
    dev_proj_out,
    d_state
    );

  cudaMemcpy(h_state, d_state, state_size, cudaMemcpyDeviceToHost);
  cudaFree(d_state);

  CUDA_CHECK_ERROR;
}
