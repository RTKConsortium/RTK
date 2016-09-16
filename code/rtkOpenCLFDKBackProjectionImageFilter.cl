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

__constant sampler_t projectionSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

__kernel void
OpenCLFDKBackProjectionImageFilterKernel(__global float *volume,
                                         __constant float *matrix,
                                         __read_only image2d_t projection,
                                         uint4 volumeDim)
{
  uint volumeIndex = get_global_id(0);

  uint i = volumeDim.x * volumeDim.y;
  uint k = volumeIndex / i;
  uint j = ( volumeIndex - (k * i) ) / volumeDim.x;
  i = volumeIndex - k * i - j * volumeDim.x;

  if (k >= volumeDim.z)
    return;

  float2 ip;
  float  ipz;

  // matrix multiply
  ip.x = matrix[0]*i + matrix[1]*j + matrix[ 2]*k + matrix[ 3];
  ip.y = matrix[4]*i + matrix[5]*j + matrix[ 6]*k + matrix[ 7];
  ipz  = matrix[8]*i + matrix[9]*j + matrix[10]*k + matrix[11];
  ipz = 1 / ipz;
  ip.x = ip.x * ipz;
  ip.y = ip.y * ipz;

  // Get projection value and add
  float4 projectionData = read_imagef(projection, projectionSampler, ip);

  volume[volumeIndex] += ipz * ipz * projectionData.x;
}

/*
__kernel
void kernel_forwardProject_noTexture(float *dev_proj_in, image2d_t *dv
                                     float *dev_proj_out,
									 float *dev_vol)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int numThread = j*c_projSize.x + i;

  if (i >= c_projSize.x || j >= c_projSize.y)
    return;

  // Declare variables used in the loop
  Ray ray;
  float3 pixelPos;
  float tnear, tfar;

  for (unsigned int proj = 0; proj<c_projSize.z; proj++)
    {
    // Setting ray origin
    ray.o = make_float3(c_sourcePos[3 * proj], c_sourcePos[3 * proj + 1], c_sourcePos[3 * proj + 2]);

    pixelPos = matrix_multiply(make_float3(i,j,0), &(c_matrices[12*proj]));

    ray.d = pixelPos - ray.o;
    ray.d = ray.d / sqrtf(dot(ray.d,ray.d));

    // Detect intersection with box
    if ( !intersectBox(ray, &tnear, &tfar, c_boxMin, c_boxMax) || tfar < 0.f )
      {
      dev_proj_out[numThread + proj * c_projSize.x * c_projSize.y] = dev_proj_in[numThread + proj * c_projSize.x * c_projSize.y];
      }
    else
      {
      if (tnear < 0.f)
        tnear = 0.f; // clamp to near plane

      // Step length in mm
      float3 dirInMM = c_spacing * ray.d;
      float vStep = c_tStep / sqrtf(dot(dirInMM, dirInMM));
      float3 step = vStep * ray.d;

      // First position in the box
      float3 pos;
      float halfVStep = 0.5f*vStep;
      tnear = tnear + halfVStep;
      pos = ray.o + tnear*ray.d;

      float  t;
      float  sample = 0.0f;
      float  sum    = 0.0f;
      for(t=tnear; t<=tfar; t+=vStep)
        {
        // Read from 3D texture from volume
        sample = notex3D(dev_vol, pos, c_volSize);

        sum += sample;
        pos += step;
        }
      dev_proj_out[numThread + proj * c_projSize.x * c_projSize.y] = dev_proj_in[numThread + proj * c_projSize.x * c_projSize.y] + (sum+(tfar-t+halfVStep)/vStep*sample) * c_tStep;
      }
    }
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_forward_project() //////////////////////////////////
void
CUDA_forward_project( int projSize[3],
                      int volSize[3],
                      float* matrices,
                      float *dev_proj_in,
                      float *dev_proj_out,
                      float *dev_vol,
                      float t_step,
                      float* source_positions,
                      float box_min[3],
                      float box_max[3],
                      float spacing[3],
                      bool useCudaTexture)
{
  // Constant memory
//  float3 dev_boxMin = make_float3(box_min[0], box_min[1], box_min[2]);
//  float3 dev_boxMax = make_float3(box_max[0], box_max[1], box_max[2]);
//  float3 dev_spacing = make_float3(spacing[0], spacing[1], spacing[2]);
  cudaMemcpyToSymbol(c_projSize, projSize, sizeof(int3));
  cudaMemcpyToSymbol(c_boxMin, box_min, sizeof(float3));
  cudaMemcpyToSymbol(c_boxMax, box_max, sizeof(float3));
  cudaMemcpyToSymbol(c_spacing, spacing, sizeof(float3));
  cudaMemcpyToSymbol(c_volSize, volSize, sizeof(int3));
  cudaMemcpyToSymbol(c_tStep, &t_step, sizeof(float));

  dim3 dimBlock  = dim3(16, 16, 1);
  dim3 dimGrid = dim3(iDivUp(projSize[0], dimBlock.x), iDivUp(projSize[1], dimBlock.x));

  // Copy the source position matrix into a float3 in constant memory
  cudaMemcpyToSymbol(c_sourcePos, &(source_positions[0]), 3 * sizeof(float) * projSize[2]);

  // Copy the projection matrices into constant memory
  cudaMemcpyToSymbol(c_matrices, &(matrices[0]), 12 * sizeof(float) * projSize[2]);

  kernel_forwardProject_noTexture<<<dimGrid, dimBlock >>> (dev_proj_in, dev_proj_out, dev_vol);
}
*/