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

#include "rtkCudaParkerShortScanImageFilter.hcu"
#include "rtkCudaUtilities.hcu"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

texture<float, 1, cudaReadModeElementType> tex_geometry; // geometry texture

inline __device__
float TransformIndexToPhysicalPoint(int2 idx, float origin, float row, float column)
{
  return origin + row * idx.x + column * idx.y;
}

inline __device__
float ToUntiltedCoordinateAtIsocenter(float tiltedCoord, float sdd, float sid, float sx, float px, float sidu)
{
  // sidu is the distance between the source and the virtual untilted detector
  // l is the coordinate on the virtual detector parallel to the real detector
  // and passing at the isocenter
  const float l = (tiltedCoord + px - sx) * sid / sdd + sx;
  // a is the angle between the virtual detector and the real detector
  const float cosa = sx / sidu;
  // the following relation refers to a note by R. Clackdoyle, title
  // "Samping a tilted detector"
  return l * sid / (sidu - l * cosa);
}

__global__
void kernel_parker_weight(
  int2 proj_idx,
  int3 proj_size,
  int2 proj_size_buf_in,
  int2 proj_size_buf_out,
  float *dev_proj_in,
  float *dev_proj_out,
  float delta,
  float firstAngle,
  float proj_orig,    // projection origin
  float proj_row,     // projection row direction & spacing
  float proj_col      // projection col direction & spacing
)
{
  // compute projection index (== thread index)
  int3 pIdx;
  pIdx.x = blockIdx.x * blockDim.x + threadIdx.x;
  pIdx.y = blockIdx.y * blockDim.y + threadIdx.y;
  pIdx.z = blockIdx.z * blockDim.z + threadIdx.z;
  long int pIdx_comp_in = pIdx.x + (pIdx.y + pIdx.z * proj_size_buf_in.y)*(proj_size_buf_in.x);
  long int pIdx_comp_out = pIdx.x + (pIdx.y + pIdx.z * proj_size_buf_out.y)*(proj_size_buf_out.x);

  // check if outside of projection grid
  if (pIdx.x >= proj_size.x || pIdx.y >= proj_size.y || pIdx.z >= proj_size.z)
    return;

  float sdd = tex1Dfetch(tex_geometry, pIdx.z * 5 + 0);
  float sx = tex1Dfetch(tex_geometry, pIdx.z * 5 + 1);
  float px = tex1Dfetch(tex_geometry, pIdx.z * 5 + 2);
  float sid = tex1Dfetch(tex_geometry, pIdx.z * 5 + 3);

  // convert actual index to point
  float pPoint = TransformIndexToPhysicalPoint(
        make_int2(pIdx.x + proj_idx.x, pIdx.y + proj_idx.y), proj_orig, proj_row, proj_col);

  // alpha projection angle
  float hyp = sqrtf(sid * sid + sx * sx); // to untilted situation
  float invsid = 1.f / hyp;
  float l = ToUntiltedCoordinateAtIsocenter(pPoint, sdd, sid, sx, px, hyp);
  float alpha = atan(-1 * l * invsid);

  // beta projection angle: Parker's article assumes that the scan starts at 0
  float beta = tex1Dfetch(tex_geometry, pIdx.z * 5 + 4);
  beta -= firstAngle;
  if (beta < 0)
    beta += (2.f * CUDART_PI_F);

  // compute weight
  float weight = 0.;
  if (beta <= (2 * delta - 2 * alpha))
    weight = 2.f * powf(
          sinf((CUDART_PI_F * beta) / (4 * (delta - alpha))),
          2.f);
  else if (beta <= (CUDART_PI_F - 2 * alpha))
    weight = 2.f;
  else if (beta <= (CUDART_PI_F + 2 * delta))
    weight = 2.f * powf(
          sinf((CUDART_PI_F * (CUDART_PI_F + 2 * delta - beta) ) / (4 * (delta + alpha))),
          2.f);

  // compute outpout by multiplying with weight
  dev_proj_out[pIdx_comp_out] = dev_proj_in[pIdx_comp_in] * weight;
}

void
CUDA_parker_weight(
  int proj_idx[2],
  int proj_dim[3],
  int proj_dim_buf_in[2],
  int proj_dim_buf_out[2],
  float *dev_proj_in,
  float *dev_proj_out,
  float *geometries,
  float delta,
  float firstAngle,
  float proj_orig,
  float proj_row,
  float proj_col)
{
  // copy geometry matrix to device, bind the matrix to the texture
  float *dev_geom;
  cudaMalloc((void**)&dev_geom, proj_dim[2]*5*sizeof(float));
  CUDA_CHECK_ERROR;
  cudaMemcpy(dev_geom, geometries, proj_dim[2]*5*sizeof(float), cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR;
  cudaBindTexture(0, tex_geometry, dev_geom, proj_dim[2]*5*sizeof(float));
  CUDA_CHECK_ERROR;

  // Thread Block Dimensions
  int tBlock_x = 16;
  int tBlock_y = 16;
  int tBlock_z = 2;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int  blocksInX = (proj_dim[0] - 1) / tBlock_x + 1;
  unsigned int  blocksInY = (proj_dim[1] - 1) / tBlock_y + 1;
  unsigned int  blocksInZ = (proj_dim[2] - 1) / tBlock_z + 1;

  dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
  kernel_parker_weight <<< dimGrid, dimBlock >>> (
      make_int2(proj_idx[0], proj_idx[1]),
      make_int3(proj_dim[0], proj_dim[1], proj_dim[2]),
      make_int2(proj_dim_buf_in[0], proj_dim_buf_in[1]),
      make_int2(proj_dim_buf_out[0], proj_dim_buf_out[1]),
      dev_proj_in,
      dev_proj_out,
      delta, firstAngle,
      proj_orig,
      proj_row,
      proj_col
      );

  // Unbind matrix texture
  cudaUnbindTexture(tex_geometry);
  CUDA_CHECK_ERROR;
  cudaFree(dev_geom);
  CUDA_CHECK_ERROR;
}
