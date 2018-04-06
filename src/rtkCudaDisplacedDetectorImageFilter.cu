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

#include "rtkCudaDisplacedDetectorImageFilter.hcu"
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
void kernel_displaced_weight(
  int3 proj_idx_in,
  int3 proj_size_in,
  int2 proj_size_in_buf,
  int3 proj_idx_out,
  int3 proj_size_out,
  int2 proj_size_out_buf,
  float *dev_proj_in,
  float *dev_proj_out,
  float theta,
  bool isPositiveCase,
  float proj_orig,    // projection origin
  float proj_row,     // projection row direction & spacing
  float proj_col      // projection col direction & spacing
)
{
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
  long int pIdx_comp = (pIdx.x-proj_idx_in.x) + (pIdx.y-proj_idx_in.y) * proj_size_in_buf.x + (pIdx.z-proj_idx_in.z) * proj_size_in_buf.x * proj_size_in_buf.y;

  // check if outside overlapping region
  if (pIdx.x < proj_idx_in.x || pIdx.x >= (proj_idx_in.x + proj_size_in.x) ||
      pIdx.y < proj_idx_in.y || pIdx.y >= (proj_idx_in.y + proj_size_in.y) ||
      pIdx.z < proj_idx_in.z || pIdx.z >= (proj_idx_in.z + proj_size_in.z))
  {
    // set areas outside overlapping region to zero
    dev_proj_out[tIdx_comp] = 0.f;
    return;
  }
  else
  {
    float pPoint = TransformIndexToPhysicalPoint(
          make_int2(pIdx.x, pIdx.y), proj_orig, proj_row, proj_col);

    float sdd = tex1Dfetch(tex_geometry, tIdx.z * 4 + 0);
    float sx = tex1Dfetch(tex_geometry, tIdx.z * 4 + 1);
    float px   = tex1Dfetch(tex_geometry, tIdx.z * 4 + 2);
    float sid = tex1Dfetch(tex_geometry, tIdx.z * 4 + 3);

    float hyp = sqrtf(sid * sid + sx * sx); // to untilted situation
    float l = ToUntiltedCoordinateAtIsocenter(pPoint, sdd, sid, sx, px, hyp);
    float invsdd = 0.f;
    float invden = 0.f;
    if (hyp != 0.f)
    {
      invsdd = 1.f / hyp;
      invden = 1.f / (2.f * atanf(theta * invsdd));
    }

    // compute weights here
    float weight = 0.f;
    if (isPositiveCase)
    {
      if (l <= -1.f * theta)
        weight = 0.f;
      else if (l >= theta)
        weight = 2.f;
      else
        weight = sinf(CUDART_PI_F * atanf(l * invsdd) * invden) + 1.f;
    }
    else
    {
      if (l <= -1.f * theta)
        weight = 2.f;
      else if (l >= theta)
        weight = 0.f;
      else
        weight = 1.f - sinf(CUDART_PI_F * atanf(l * invsdd) * invden);
    }

    dev_proj_out[tIdx_comp] = dev_proj_in[pIdx_comp] * weight;
  }
}

void
CUDA_displaced_weight(
  int proj_idx_in[3], // overlapping input region index
  int proj_dim_in[3], // overlapping input region size
  int proj_dim_in_buf[2], // input size of buffered region
  int proj_idx_out[3], // output region index
  int proj_dim_out[3], // output region size
  int proj_dim_out_buf[2], // output size of buffered region
  float *dev_proj_in,
  float *dev_proj_out,
  float *geometries,
  float theta,
  bool isPositiveCase,
  float proj_orig,
  float proj_row,
  float proj_col)
{
  // copy geometry matrix to device, bind the matrix to the texture
  float *dev_geom;
  cudaMalloc((void**)&dev_geom, proj_dim_out[2]*4*sizeof(float));
  CUDA_CHECK_ERROR;
  cudaMemcpy(dev_geom, geometries, proj_dim_out[2]*4*sizeof(float), cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR;
  cudaBindTexture(0, tex_geometry, dev_geom, proj_dim_out[2]*4*sizeof(float));
  CUDA_CHECK_ERROR;

  // Thread Block Dimensions
  int tBlock_x = 16;
  int tBlock_y = 16;
  int tBlock_z = 2;

  // Each element in the volume (each voxel) gets 1 thread
  unsigned int  blocksInX = (proj_dim_out[0] - 1) / tBlock_x + 1;
  unsigned int  blocksInY = (proj_dim_out[1] - 1) / tBlock_y + 1;
  unsigned int  blocksInZ = (proj_dim_out[2] - 1) / tBlock_z + 1;

  dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
  kernel_displaced_weight <<< dimGrid, dimBlock >>> (
      make_int3(proj_idx_in[0], proj_idx_in[1], proj_idx_in[2]),
      make_int3(proj_dim_in[0], proj_dim_in[1], proj_dim_in[2]),
      make_int2(proj_dim_in_buf[0], proj_dim_in_buf[1]),
      make_int3(proj_idx_out[0], proj_idx_out[1], proj_idx_out[2]),
      make_int3(proj_dim_out[0], proj_dim_out[1], proj_dim_out[2]),
      make_int2(proj_dim_out_buf[0], proj_dim_out_buf[1]),
      dev_proj_in,
      dev_proj_out,
      theta, isPositiveCase,
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
