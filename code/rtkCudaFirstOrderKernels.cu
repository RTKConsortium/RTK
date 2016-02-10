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

#include "rtkCudaFirstOrderKernels.hcu"

__global__ void divergence_kernel(float * grad_x, float * grad_y, float * grad_z, float * out, int3 c_Size, float3 c_Spacing)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= c_Size.x || j >= c_Size.y  || k >= c_Size.z)
      return;

  long int id   = (k     * c_Size.y + j)    * c_Size.x + i;
  long int id_x = (k     * c_Size.y + j)    * c_Size.x + i - 1;
  long int id_y = (k     * c_Size.y + j - 1)* c_Size.x + i;
  long int id_z = ((k-1) * c_Size.y + j)    * c_Size.x + i;

  float3 A;
  float3 B;

  if (i == 0) B.x = 0;
  else B.x = grad_x[id_x];
  if (i == (c_Size.x - 1)) A.x = 0;
  else A.x = grad_x[id];

  if (j == 0) B.y = 0;
  else B.y = grad_y[id_y];
  if (j == (c_Size.y - 1)) A.y = 0;
  else A.y = grad_y[id];

  if (k == 0) B.z = 0;
  else B.z = grad_z[id_z];
  if (k == (c_Size.z - 1)) A.z = 0;
  else A.z = grad_z[id];

  out[id] = (A.x - B.x) / c_Spacing.x
          + (A.y - B.y) / c_Spacing.y
          + (A.z - B.z) / c_Spacing.z;
}

__global__ void gradient_kernel(float * in, float * grad_x, float * grad_y, float * grad_z, int3 c_Size, float3 c_Spacing)
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

  if (i == (c_Size.x - 1)) grad_x[id] = 0;
  else grad_x[id] = (in[id_x] - in[id]) / c_Spacing.x;

  if (j == (c_Size.y - 1)) grad_y[id] = 0;
  else grad_y[id] = (in[id_y] - in[id]) / c_Spacing.y;

  if (k == (c_Size.z - 1)) grad_z[id] = 0;
  else grad_z[id] = (in[id_z] - in[id]) / c_Spacing.z;
}
