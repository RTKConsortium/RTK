//
#include "rtkCudaFDKWeightProjectionFilter.hcu"
#include "rtkCudaUtilities.hcu"
#include <cuda.h>
#include <cuda_runtime.h>

texture<float, 1, cudaReadModeElementType> tex_geometry; // geometry texture

inline __device__
float3 TransformIndexToPhysicalPoint(int2 idx, float3 origin, float3 row, float3 column)
{
  return make_float3(
        origin.x + row.x * idx.x + column.x * idx.y,
        origin.y + row.y * idx.x + column.y * idx.y,
        origin.z + row.z * idx.x + column.z * idx.y
        );
}

__global__
void kernel_weight_projection(
  int3 proj_size,
  float *dev_proj_in,
  float *dev_proj_out,
  float3 proj_orig,    // projection origin
  float3 proj_row,     // projection row direction & spacing
  float3 proj_col      // projection col direction & spacing
)
{
  // compute projection index (== thread index)
  int3 pIdx;
  pIdx.x = blockIdx.x * blockDim.x + threadIdx.x;
  pIdx.y = blockIdx.y * blockDim.y + threadIdx.y;
  pIdx.z = blockIdx.z * blockDim.z + threadIdx.z;
  long int pIdx_comp = pIdx.x + (pIdx.y + pIdx.z * proj_size.y)*(proj_size.x);

  // check if outside of projection grid
  if (pIdx.x >= proj_size.x || pIdx.y >= proj_size.y || pIdx.z >= proj_size.z)
    return;

  const float sdd = tex1Dfetch(tex_geometry, pIdx.z * 7 + 0);
  const float sid = tex1Dfetch(tex_geometry, pIdx.z * 7 + 1);
  const float wFac = tex1Dfetch(tex_geometry, pIdx.z * 7 + 5);
  if (sdd == 0) // parallel
  {
    dev_proj_out[pIdx_comp] = dev_proj_in[pIdx_comp] * wFac;
  }
  else // divergent
  {
    const float pOffX = tex1Dfetch(tex_geometry, pIdx.z * 7 + 2);
    const float pOffY = tex1Dfetch(tex_geometry, pIdx.z * 7 + 3);
    const float sOffY = tex1Dfetch(tex_geometry, pIdx.z * 7 + 4);
    const float tAngle = tex1Dfetch(tex_geometry, pIdx.z * 7 + 6);
    const float sina = sin(tAngle);
    const float cosa = cos(tAngle);
    const float tana = tan(tAngle);

    // compute projection point from index
    float3 pPoint = TransformIndexToPhysicalPoint(
          make_int2(pIdx.x, pIdx.y), proj_orig, proj_row, proj_col);
    pPoint.x = pPoint.x + pOffX + tana * (sdd - sid);
    pPoint.y = pPoint.y + pOffY - sOffY;

    const float numpart1 = sdd*(cosa+tana*sina);
    const float denom = sqrt((sdd * sdd + pPoint.y * pPoint.y) +
                             ((pPoint.x - sdd * tana) * (pPoint.x - sdd * tana)));
    const float cosGamma = (numpart1 - pPoint.x * sina) / denom;
    dev_proj_out[pIdx_comp] = dev_proj_in[pIdx_comp] * wFac * cosGamma;
  }
}

void
CUDA_weight_projection(
  int proj_dim[3],
  float *dev_proj_in,
  float *dev_proj_out,
  float *geometries,
  float proj_orig[3],
  float proj_row [3],
  float proj_col[3]
)
{
  // copy geometry matrix to device, bind the matrix to the texture
  float *dev_geom;
  cudaMalloc((void**)&dev_geom, proj_dim[2]*7*sizeof(float));
  cudaMemcpy(dev_geom, geometries, proj_dim[2]*7*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTexture(0, tex_geometry, dev_geom, proj_dim[2]*7*sizeof(float));

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
  kernel_weight_projection <<< dimGrid, dimBlock >>> (
      make_int3(proj_dim[0], proj_dim[1], proj_dim[2]),
      dev_proj_in,
      dev_proj_out,
      make_float3(proj_orig[0], proj_orig[1], proj_orig[2]),
      make_float3(proj_row[0], proj_row[1], proj_row[2]),
      make_float3(proj_col[0], proj_col[1], proj_col[2])
      );

  // Unbind matrix texture
  cudaUnbindTexture(tex_geometry);
  cudaFree(dev_geom);
  CUDA_CHECK_ERROR;
}
