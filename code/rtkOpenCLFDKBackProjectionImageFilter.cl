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

