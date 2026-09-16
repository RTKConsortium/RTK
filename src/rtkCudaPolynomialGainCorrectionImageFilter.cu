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

#include "rtkCudaPolynomialGainCorrectionImageFilter.hcu"
#include "rtkCudaUtilities.hcu"
#include <itkIntTypes.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// System-adaptive buffer index types (ITK)
using SizeValueType = itk::SizeValueType;

__constant__ float cst_coef[2];

__global__ void
kernel_gain_correction(int3             proj_idx_in,
                       int3             proj_size_in,
                       int2             proj_size_in_buf,
                       int3             proj_idx_out,
                       int3             proj_size_out,
                       int2             proj_size_out_buf,
                       unsigned short * dev_proj_in,
                       float *          dev_proj_out,
                       unsigned short * dev_dark_in,
                       float *          dev_gain_in,
                       float *          powerlut)
{
  // compute thread index (64-bit to avoid 32-bit overflow of combined indices)
  SizeValueType tIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
  SizeValueType tIdx_y = blockIdx.y * blockDim.y + threadIdx.y;
  SizeValueType tIdx_z = blockIdx.z * blockDim.z + threadIdx.z;
  SizeValueType tIdx_comp = tIdx_x + tIdx_y * proj_size_out.x + tIdx_z * proj_size_out_buf.x * proj_size_out_buf.y;

  // check if outside of projection grid
  if (tIdx_x >= proj_size_out.x || tIdx_y >= proj_size_out.y || tIdx_z >= proj_size_out.z)
    return;

  // combined projection index (arithmetic in 64-bit to avoid overflow of the products)
  SizeValueType pIdx_comp = (tIdx_x - proj_idx_in.x + proj_idx_out.x) +
                            (tIdx_y - proj_idx_in.y + proj_idx_out.y) * proj_size_in_buf.x +
                            (tIdx_z - proj_idx_in.z + proj_idx_out.z) * proj_size_in_buf.x * proj_size_in_buf.y;

  int modelOrder = static_cast<float>(cst_coef[0]);

  SizeValueType sIdx_comp = tIdx_x + tIdx_y * proj_size_out.x; // in-slice index

  // Correct for dark field
  unsigned short xk = 0;
  if (dev_proj_in[pIdx_comp] > dev_dark_in[sIdx_comp])
    xk = dev_proj_in[pIdx_comp] - dev_dark_in[sIdx_comp];

  float         yk = 0.f;
  SizeValueType lutidx = static_cast<SizeValueType>(xk) * modelOrder; // index to powerlut
  SizeValueType projsize = static_cast<SizeValueType>(proj_size_in.x) * proj_size_in.y;
  for (int n = 0; n < modelOrder; n++)
  {
    SizeValueType gainidx = n * projsize + sIdx_comp;
    float         gainM = dev_gain_in[gainidx];
    yk += gainM * powerlut[lutidx + n];
  }

  // Apply normalization factor
  yk = yk * cst_coef[1];

  // Avoid negative values
  yk = (yk < 0.0f) ? 0.f : yk;

  dev_proj_out[tIdx_comp] = yk;
}

void
CUDA_gain_correction(int              proj_idx_in[3],      // overlapping input region index
                     int              proj_dim_in[3],      // overlapping input region size
                     int              proj_dim_in_buf[2],  // input size of buffered region
                     int              proj_idx_out[3],     // output region index
                     int              proj_dim_out[3],     // output region size
                     int              proj_dim_out_buf[2], // output size of buffered region
                     unsigned short * dev_proj_in,
                     float *          dev_proj_out,
                     unsigned short * dev_dark_in,
                     float *          dev_gain_in,
                     float *          h_powerlut,
                     size_t           lut_size,
                     float *          coefficients)
{
  // Thread Block Dimensions
  int tBlock_x = 16;
  int tBlock_y = 16;
  int tBlock_z = 2;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int blocksInX = (proj_dim_out[0] - 1) / tBlock_x + 1;
  unsigned int blocksInY = (proj_dim_out[1] - 1) / tBlock_y + 1;
  unsigned int blocksInZ = (proj_dim_out[2] - 1) / tBlock_z + 1;

  float * d_powerlut; // device state
  cudaMalloc((void **)&d_powerlut, lut_size);
  if (cudaMemcpy(d_powerlut, h_powerlut, lut_size, cudaMemcpyHostToDevice) != cudaSuccess)
  {
    itkGenericExceptionMacro("Error allocating state");
  }

  cudaMemcpyToSymbol(cst_coef, coefficients, 2 * sizeof(float));

  dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
  kernel_gain_correction<<<dimGrid, dimBlock>>>(make_int3(proj_idx_in[0], proj_idx_in[1], proj_idx_in[2]),
                                                make_int3(proj_dim_in[0], proj_dim_in[1], proj_dim_in[2]),
                                                make_int2(proj_dim_in_buf[0], proj_dim_in_buf[1]),
                                                make_int3(proj_idx_out[0], proj_idx_out[1], proj_idx_out[2]),
                                                make_int3(proj_dim_out[0], proj_dim_out[1], proj_dim_out[2]),
                                                make_int2(proj_dim_out_buf[0], proj_dim_out_buf[1]),
                                                dev_proj_in,
                                                dev_proj_out,
                                                dev_dark_in,
                                                dev_gain_in,
                                                d_powerlut);

  cudaFree(d_powerlut);

  CUDA_CHECK_ERROR;
}
