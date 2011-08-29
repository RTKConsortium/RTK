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

