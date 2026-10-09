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
#include "rtkCudaZengProjectionImageFilter.hcu"
#include "rtkCudaUtilities.hcu"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <map>
#include <tuple>
#include <vector>

namespace
{
constexpr int CoefficientStride = 33;
constexpr int MaximumRadius = 32;

inline __device__ float3
applyMatrix(const float * m, float x, float y, float z)
{
  return make_float3(m[0] * x + m[1] * y + m[2] * z + m[3],
                     m[4] * x + m[5] * y + m[6] * z + m[7],
                     m[8] * x + m[9] * y + m[10] * z + m[11]);
}

__device__ float
trilinearZero(const float * image, int3 size, float3 p)
{
  // Match ITK's linear interpolator at the half-voxel image boundary.
  if (!(p.x >= -0.5f && p.x < size.x - 0.5f && p.y >= -0.5f && p.y < size.y - 0.5f && p.z >= -0.5f &&
        p.z < size.z - 0.5f))
    return 0.f;
  p.x = fminf(fmaxf(p.x, 0.f), static_cast<float>(size.x - 1));
  p.y = fminf(fmaxf(p.y, 0.f), static_cast<float>(size.y - 1));
  p.z = fminf(fmaxf(p.z, 0.f), static_cast<float>(size.z - 1));
  const int   x0 = static_cast<int>(floorf(p.x));
  const int   y0 = static_cast<int>(floorf(p.y));
  const int   z0 = static_cast<int>(floorf(p.z));
  const float fx = p.x - x0;
  const float fy = p.y - y0;
  const float fz = p.z - z0;
  float       result = 0.f;
  for (int dz = 0; dz <= 1; ++dz)
    for (int dy = 0; dy <= 1; ++dy)
      for (int dx = 0; dx <= 1; ++dx)
      {
        const int x = x0 + dx;
        const int y = y0 + dy;
        const int z = z0 + dz;
        if (x >= 0 && x < size.x && y >= 0 && y < size.y && z >= 0 && z < size.z)
        {
          const float wx = dx ? fx : 1.f - fx;
          const float wy = dy ? fy : 1.f - fy;
          const float wz = dz ? fz : 1.f - fz;
          result += wx * wy * wz * image[(z * size.y + y) * size.x + x];
        }
      }
  return result;
}

std::vector<float>
discreteGaussianCoefficients(double variance)
{
  if (variance <= 1.e-12)
    return { 1.f };
  constexpr double    maximumError = 1.e-5;
  const double        exponential = std::exp(-variance);
  std::vector<double> coefficients{ exponential * std::cyl_bessel_i(0., variance) };
  double              sum = coefficients[0];
  for (int order = 1; sum < 1. - maximumError; ++order)
  {
    const double coefficient = exponential * std::cyl_bessel_i(static_cast<double>(order), variance);
    coefficients.push_back(coefficient);
    sum += 2. * coefficient;
    if (coefficient <= 0. || coefficients.size() > MaximumRadius)
      break;
  }
  std::vector<float> normalized(coefficients.size());
  std::transform(coefficients.begin(), coefficients.end(), normalized.begin(), [sum](double value) {
    return static_cast<float>(value / sum);
  });
  return normalized;
}

struct HostGaussianMetadata
{
  std::vector<float> coefficientsX;
  std::vector<float> coefficientsY;
  std::vector<int>   radiiX;
  std::vector<int>   radiiY;
};

struct MetadataKey
{
  int                mode{};
  std::vector<int>   integers;
  std::vector<float> values;

  bool
  operator<(const MetadataKey & other) const
  {
    return std::tie(mode, integers, values) < std::tie(other.mode, other.integers, other.values);
  }
};

MetadataKey
makeMetadataKey(int           mode,
                const int     projectionSize[3],
                const int     volumeSize[3],
                const int     rotatedSize[3],
                const float   rotatedSpacing[3],
                float         sigmaZero,
                float         alpha,
                const float * matrices,
                const float * distances,
                const int *   firstSlices)
{
  MetadataKey key;
  key.mode = mode;
  key.integers.insert(key.integers.end(), projectionSize, projectionSize + 3);
  key.integers.insert(key.integers.end(), volumeSize, volumeSize + 3);
  key.integers.insert(key.integers.end(), rotatedSize, rotatedSize + 3);
  key.integers.insert(key.integers.end(), firstSlices, firstSlices + projectionSize[2]);
  key.values.insert(key.values.end(), rotatedSpacing, rotatedSpacing + 3);
  key.values.push_back(sigmaZero);
  key.values.push_back(alpha);
  key.values.insert(key.values.end(), matrices, matrices + 12 * projectionSize[2]);
  key.values.insert(key.values.end(), distances, distances + projectionSize[2]);
  return key;
}

void
appendGaussian(HostGaussianMetadata &                 metadata,
               std::map<double, std::vector<float>> & cache,
               float                                  variance,
               float                                  spacingX,
               float                                  spacingY)
{
  const double         variances[] = { variance / (spacingX * spacingX), variance / (spacingY * spacingY) };
  std::vector<float> * packed[] = { &metadata.coefficientsX, &metadata.coefficientsY };
  std::vector<int> *   radii[] = { &metadata.radiiX, &metadata.radiiY };
  for (int dimension = 0; dimension < 2; ++dimension)
  {
    auto [it, inserted] = cache.try_emplace(variances[dimension]);
    if (inserted)
      it->second = discreteGaussianCoefficients(variances[dimension]);
    radii[dimension]->push_back(static_cast<int>(it->second.size()) - 1);
    packed[dimension]->insert(packed[dimension]->end(), it->second.begin(), it->second.end());
    packed[dimension]->resize(packed[dimension]->size() + CoefficientStride - it->second.size(), 0.f);
  }
}

struct DeviceMetadata
{
  float * coefficientsX{};
  float * coefficientsY{};
  float * matrices{};
  float * inverseMatrices{};
  int *   radiiX{};
  int *   radiiY{};
  int *   firstSlices{};

  ~DeviceMetadata()
  {
    cudaFree(firstSlices);
    cudaFree(radiiY);
    cudaFree(radiiX);
    cudaFree(inverseMatrices);
    cudaFree(matrices);
    cudaFree(coefficientsY);
    cudaFree(coefficientsX);
  }
};

struct Workspace
{
  float *                               current{};
  float *                               blurred{};
  float *                               scratch{};
  float *                               rotated{};
  size_t                                sliceCapacity{};
  size_t                                rotatedCapacity{};
  std::map<MetadataKey, DeviceMetadata> metadata;

  ~Workspace()
  {
    cudaFree(rotated);
    cudaFree(scratch);
    cudaFree(blurred);
    cudaFree(current);
  }
};

void
releaseSlices(Workspace & workspace)
{
  cudaFree(workspace.scratch);
  cudaFree(workspace.blurred);
  cudaFree(workspace.current);
  workspace.scratch = nullptr;
  workspace.blurred = nullptr;
  workspace.current = nullptr;
  workspace.sliceCapacity = 0;
}

bool
allocateBuffer(float ** buffer, size_t elements)
{
  const auto error = cudaMalloc(buffer, elements * sizeof(float));
  if (error == cudaErrorMemoryAllocation)
  {
    cudaGetLastError();
    return false;
  }
  if (error != cudaSuccess)
    itkGenericExceptionMacro(<< "CUDA Zeng allocation failed: " << cudaGetErrorString(error));
  return true;
}

bool
ensureSlices(Workspace & workspace, size_t elements)
{
  if (elements <= workspace.sliceCapacity)
    return true;
  releaseSlices(workspace);
  if (!allocateBuffer(&workspace.current, elements) || !allocateBuffer(&workspace.blurred, elements) ||
      !allocateBuffer(&workspace.scratch, elements))
  {
    releaseSlices(workspace);
    return false;
  }
  workspace.sliceCapacity = elements;
  return true;
}

bool
ensureRotated(Workspace & workspace, size_t elements)
{
  if (elements <= workspace.rotatedCapacity)
    return true;
  cudaFree(workspace.rotated);
  workspace.rotated = nullptr;
  workspace.rotatedCapacity = 0;
  if (!allocateBuffer(&workspace.rotated, elements))
    return false;
  workspace.rotatedCapacity = elements;
  return true;
}

unsigned int
automaticBatchSize(const Workspace & workspace, int projections, int depth, size_t pixels, bool backward)
{
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  cudaMemGetInfo(&freeBytes, &totalBytes);
  (void)totalBytes;
  const size_t slices = backward ? static_cast<size_t>(depth) + 3 : 3;
  const size_t bytesPerProjection = std::max<size_t>(1, slices * pixels * sizeof(float));
  size_t       existingCapacity = workspace.sliceCapacity / pixels;
  if (backward)
    existingCapacity = std::min(existingCapacity, workspace.rotatedCapacity / (static_cast<size_t>(depth) * pixels));
  return static_cast<unsigned int>(std::max<size_t>(
    1, std::min<size_t>(projections, std::max(freeBytes * 3 / 5 / bytesPerProjection, existingCapacity))));
}

unsigned int
allocateBatch(Workspace & workspace, unsigned int batchSize, int depth, size_t pixels, bool backward)
{
  for (;;)
  {
    if (ensureSlices(workspace, static_cast<size_t>(batchSize) * pixels) &&
        (!backward || ensureRotated(workspace, static_cast<size_t>(batchSize) * depth * pixels)))
      return batchSize;
    releaseSlices(workspace);
    if (batchSize == 1)
      itkGenericExceptionMacro(<< "Insufficient GPU memory for one CUDA Zeng projection.");
    batchSize = std::max(1u, batchSize / 2);
  }
}

void
uploadMetadata(DeviceMetadata &           device,
               HostGaussianMetadata &&    metadata,
               const float *              matrices,
               const std::vector<float> & inverseMatrices,
               const std::vector<int> &   firstSlices,
               int                        projections)
{
  cudaMalloc(&device.coefficientsX, metadata.coefficientsX.size() * sizeof(float));
  cudaMalloc(&device.coefficientsY, metadata.coefficientsY.size() * sizeof(float));
  cudaMalloc(&device.radiiX, metadata.radiiX.size() * sizeof(int));
  cudaMalloc(&device.radiiY, metadata.radiiY.size() * sizeof(int));
  cudaMalloc(&device.matrices, 12 * projections * sizeof(float));
  cudaMalloc(&device.firstSlices, projections * sizeof(int));
  cudaMemcpy(device.coefficientsX,
             metadata.coefficientsX.data(),
             metadata.coefficientsX.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device.coefficientsY,
             metadata.coefficientsY.data(),
             metadata.coefficientsY.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device.radiiX, metadata.radiiX.data(), metadata.radiiX.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device.radiiY, metadata.radiiY.data(), metadata.radiiY.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device.matrices, matrices, 12 * projections * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device.firstSlices, firstSlices.data(), projections * sizeof(int), cudaMemcpyHostToDevice);
  if (!inverseMatrices.empty())
  {
    cudaMalloc(&device.inverseMatrices, inverseMatrices.size() * sizeof(float));
    cudaMemcpy(
      device.inverseMatrices, inverseMatrices.data(), inverseMatrices.size() * sizeof(float), cudaMemcpyHostToDevice);
  }
  CUDA_CHECK_ERROR;
}

__global__ void
gaussianX(const float * input,
          float *       output,
          int           width,
          int           height,
          const float * coefficients,
          const int *   radii,
          const int *   firstSlices,
          int           metadataStride,
          int           metadataZ,
          int           batchStart)
{
  extern __shared__ float tile[];
  const int               localProjection = blockIdx.z;
  if (metadataZ < firstSlices[batchStart + localProjection])
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
      const size_t offset = static_cast<size_t>(localProjection) * width * height + y * width + x;
      output[offset] = input[offset];
    }
    return;
  }
  const int    metadataIndex = (batchStart + localProjection) * metadataStride + metadataZ;
  const int    radius = radii[metadataIndex];
  const int    tileWidth = blockDim.x + 2 * radius;
  const int    tileElements = tileWidth * blockDim.y;
  const int    threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
  const int    threadCount = blockDim.x * blockDim.y;
  const size_t sliceOffset = static_cast<size_t>(localProjection) * width * height;
  for (int index = threadIndex; index < tileElements; index += threadCount)
  {
    const int localX = index % tileWidth;
    const int localY = index / tileWidth;
    const int x = blockIdx.x * blockDim.x + localX - radius;
    const int y = blockIdx.y * blockDim.y + localY;
    tile[index] = x >= 0 && x < width && y < height ? input[sliceOffset + y * width + x] : 0.f;
  }
  __syncthreads();
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  float         sum = 0.f;
  const float * kernel = coefficients + static_cast<size_t>(metadataIndex) * CoefficientStride;
  for (int offset = -radius; offset <= radius; ++offset)
    sum += kernel[abs(offset)] * tile[threadIdx.y * tileWidth + threadIdx.x + radius + offset];
  output[sliceOffset + y * width + x] = sum;
}

__global__ void
gaussianY(const float * input,
          float *       output,
          int           width,
          int           height,
          const float * coefficients,
          const int *   radii,
          const int *   firstSlices,
          int           metadataStride,
          int           metadataZ,
          int           batchStart)
{
  extern __shared__ float tile[];
  const int               localProjection = blockIdx.z;
  if (metadataZ < firstSlices[batchStart + localProjection])
  {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
      const size_t offset = static_cast<size_t>(localProjection) * width * height + y * width + x;
      output[offset] = input[offset];
    }
    return;
  }
  const int    metadataIndex = (batchStart + localProjection) * metadataStride + metadataZ;
  const int    radius = radii[metadataIndex];
  const int    tileHeight = blockDim.y + 2 * radius;
  const int    tileElements = blockDim.x * tileHeight;
  const int    threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
  const int    threadCount = blockDim.x * blockDim.y;
  const size_t sliceOffset = static_cast<size_t>(localProjection) * width * height;
  for (int index = threadIndex; index < tileElements; index += threadCount)
  {
    const int localX = index % blockDim.x;
    const int localY = index / blockDim.x;
    const int x = blockIdx.x * blockDim.x + localX;
    const int y = blockIdx.y * blockDim.y + localY - radius;
    tile[index] = x < width && y >= 0 && y < height ? input[sliceOffset + y * width + x] : 0.f;
  }
  __syncthreads();
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  float         sum = 0.f;
  const float * kernel = coefficients + static_cast<size_t>(metadataIndex) * CoefficientStride;
  for (int offset = -radius; offset <= radius; ++offset)
    sum += kernel[abs(offset)] * tile[(threadIdx.y + radius + offset) * blockDim.x + threadIdx.x];
  output[sliceOffset + y * width + x] = sum;
}

void
gaussianBatch(const float *          input,
              float *                output,
              float *                scratch,
              const DeviceMetadata & metadata,
              int                    width,
              int                    height,
              int                    batchStart,
              int                    batchCount,
              int                    metadataZ,
              int                    metadataStride)
{
  const dim3   block(16, 16);
  const dim3   grid(iDivUp(width, 16), iDivUp(height, 16), batchCount);
  const size_t sharedX = (block.x + 2 * MaximumRadius) * block.y * sizeof(float);
  const size_t sharedY = block.x * (block.y + 2 * MaximumRadius) * sizeof(float);
  gaussianX<<<grid, block, sharedX>>>(input,
                                      scratch,
                                      width,
                                      height,
                                      metadata.coefficientsX,
                                      metadata.radiiX,
                                      metadata.firstSlices,
                                      metadataStride,
                                      metadataZ,
                                      batchStart);
  gaussianY<<<grid, block, sharedY>>>(scratch,
                                      output,
                                      width,
                                      height,
                                      metadata.coefficientsY,
                                      metadata.radiiY,
                                      metadata.firstSlices,
                                      metadataStride,
                                      metadataZ,
                                      batchStart);
}

__global__ void
sampleForward(float *       current,
              const float * previous,
              const float * volume,
              const float * attenuation,
              int3          volumeSize,
              const float * matrices,
              const int *   firstSlices,
              int           width,
              int           height,
              int           z,
              float         attenuationStep,
              int           batchStart,
              bool          addPrevious)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int localProjection = blockIdx.z;
  const int projection = batchStart + localProjection;
  if (x >= width || y >= height || z < firstSlices[projection])
    return;
  const int    pixel = y * width + x;
  const size_t offset = static_cast<size_t>(localProjection) * width * height + pixel;
  const float3 position = applyMatrix(matrices + 12 * projection, x, y, z);
  float        value = trilinearZero(volume, volumeSize, position);
  if (addPrevious)
    value += previous[offset];
  if (attenuation)
    value *= expf(-attenuationStep * trilinearZero(attenuation, volumeSize, position));
  current[offset] = value;
}

__global__ void
finishForward(const float * input, const float * zeng, float * output, int pixels, float thickness, int batchStart)
{
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int localProjection = blockIdx.y;
  if (pixel >= pixels)
    return;
  const int    projection = batchStart + localProjection;
  const size_t local = static_cast<size_t>(localProjection) * pixels + pixel;
  output[static_cast<size_t>(projection) * pixels + pixel] =
    input[static_cast<size_t>(projection) * pixels + pixel] + thickness * zeng[local];
}

__global__ void
copyProjections(const float * projections, float * current, int pixels, int batchStart)
{
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int localProjection = blockIdx.y;
  if (pixel < pixels)
    current[static_cast<size_t>(localProjection) * pixels + pixel] =
      projections[static_cast<size_t>(batchStart + localProjection) * pixels + pixel];
}

__global__ void
storeSlices(const float * current,
            float *       rotated,
            int           pixels,
            int           depth,
            int           z,
            const int *   firstSlices,
            int           batchStart)
{
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int localProjection = blockIdx.y;
  const int projection = batchStart + localProjection;
  if (pixel < pixels && z >= firstSlices[projection])
    rotated[(static_cast<size_t>(localProjection) * depth + z) * pixels + pixel] =
      current[static_cast<size_t>(localProjection) * pixels + pixel];
}

__global__ void
attenuate(float *             current,
          cudaTextureObject_t attenuation,
          int3                volumeSize,
          const float *       inverseMatrices,
          int                 width,
          int                 height,
          int                 z,
          const int *         firstSlices,
          bool                firstSlice,
          float               step,
          int                 batchStart)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int localProjection = blockIdx.z;
  const int projection = batchStart + localProjection;
  if (x >= width || y >= height || (!firstSlice && z <= firstSlices[projection]))
    return;
  const int    slice = firstSlice ? firstSlices[projection] : z;
  const float3 position = applyMatrix(inverseMatrices + 12 * projection, x, y, slice);
  const size_t offset = static_cast<size_t>(localProjection) * width * height + y * width + x;
  if (position.x >= 0.f && position.x < volumeSize.x && position.y >= 0.f && position.y < volumeSize.y &&
      position.z >= 0.f && position.z < volumeSize.z)
    current[offset] *= expf(-step * tex3D<float>(attenuation, position.x, position.y, position.z));
}

__global__ void
addRotatedBatch(const float * rotated,
                float *       output,
                int3          volumeSize,
                int3          rotatedSize,
                const float * matrices,
                int           batchStart,
                int           batchCount,
                float         thickness)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= volumeSize.x || y >= volumeSize.y || z >= volumeSize.z)
    return;
  float        sum = 0.f;
  const size_t rotatedElements = static_cast<size_t>(rotatedSize.x) * rotatedSize.y * rotatedSize.z;
  for (int localProjection = 0; localProjection < batchCount; ++localProjection)
  {
    const int    projection = batchStart + localProjection;
    const float3 position = applyMatrix(matrices + 12 * projection, x, y, z);
    sum += trilinearZero(rotated + localProjection * rotatedElements, rotatedSize, position);
  }
  const int index = (z * volumeSize.y + y) * volumeSize.x + x;
  output[index] += thickness * sum;
}

void
invertAffine(const float * source, float * inverse)
{
  const double a = source[0], b = source[1], c = source[2];
  const double d = source[4], e = source[5], f = source[6];
  const double g = source[8], h = source[9], i = source[10];
  const double determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  const double inv[9] = { (e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant,
                          (f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant,
                          (d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant };
  for (int r = 0; r < 3; ++r)
  {
    for (int col = 0; col < 3; ++col)
      inverse[4 * r + col] = static_cast<float>(inv[3 * r + col]);
    inverse[4 * r + 3] =
      static_cast<float>(-(inv[3 * r] * source[3] + inv[3 * r + 1] * source[7] + inv[3 * r + 2] * source[11]) + 0.5);
  }
}
} // namespace

void
CUDA_zeng_forward_project(const int     projectionSize[3],
                          const int     volumeSize[3],
                          const int     rotatedSize[3],
                          const float   rotatedSpacing[3],
                          const float * rotatedToVolumeMatrices,
                          const float * farDistances,
                          const float * devProjectionIn,
                          float *       devProjectionOut,
                          const float * devVolume,
                          const float * devAttenuation,
                          float         sigmaZero,
                          float         alpha,
                          unsigned int  requestedBatchSize,
                          void **       workspacePointer)
{
  if (!*workspacePointer)
    *workspacePointer = new Workspace;
  auto &             workspace = *static_cast<Workspace *>(*workspacePointer);
  const int          projections = projectionSize[2];
  const int          depth = rotatedSize[2];
  const int          pixels = rotatedSize[0] * rotatedSize[1];
  const int          metadataStride = depth + 1;
  std::vector<int>   firstSlices(projections);
  std::vector<float> nearDistances(projections);
  for (int projection = 0; projection < projections; ++projection)
  {
    float nearDistance = farDistances[projection];
    int   first = depth - 1;
    while (first > 0 && nearDistance - rotatedSpacing[2] >= 0.f)
    {
      --first;
      nearDistance -= rotatedSpacing[2];
    }
    firstSlices[projection] = first;
    nearDistances[projection] = nearDistance;
  }
  const auto key = makeMetadataKey(0,
                                   projectionSize,
                                   volumeSize,
                                   rotatedSize,
                                   rotatedSpacing,
                                   sigmaZero,
                                   alpha,
                                   rotatedToVolumeMatrices,
                                   farDistances,
                                   firstSlices.data());
  auto [iterator, inserted] = workspace.metadata.try_emplace(key);
  auto & deviceMetadata = iterator->second;
  if (inserted)
  {
    HostGaussianMetadata                 metadata;
    std::map<double, std::vector<float>> cache;
    for (int projection = 0; projection < projections; ++projection)
    {
      const float nearDistance = nearDistances[projection];
      for (int z = 0; z < depth; ++z)
      {
        float variance = 0.f;
        if (z >= firstSlices[projection] && z + 1 < depth)
        {
          const float distance = nearDistance + (z + 1 - firstSlices[projection]) * rotatedSpacing[2];
          variance = distance * 2.f * rotatedSpacing[2] * alpha * alpha + 2.f * rotatedSpacing[2] * alpha * sigmaZero -
                     alpha * alpha * rotatedSpacing[2] * rotatedSpacing[2];
        }
        appendGaussian(metadata, cache, std::max(0.f, variance), rotatedSpacing[0], rotatedSpacing[1]);
      }
      const float finalVariance = (alpha * nearDistance + sigmaZero) * (alpha * nearDistance + sigmaZero);
      appendGaussian(metadata, cache, finalVariance, rotatedSpacing[0], rotatedSpacing[1]);
    }
    uploadMetadata(deviceMetadata, std::move(metadata), rotatedToVolumeMatrices, {}, firstSlices, projections);
  }

  const unsigned int desiredBatchSize = requestedBatchSize
                                          ? std::min(requestedBatchSize, static_cast<unsigned int>(projections))
                                          : automaticBatchSize(workspace, projections, depth, pixels, false);
  const unsigned int batchSize = allocateBatch(workspace, desiredBatchSize, depth, pixels, false);
  const dim3         block(16, 16);
  const int3         cudaVolumeSize = make_int3(volumeSize[0], volumeSize[1], volumeSize[2]);
  for (int batchStart = 0; batchStart < projections; batchStart += batchSize)
  {
    const int  batchCount = std::min<int>(batchSize, projections - batchStart);
    const dim3 grid(iDivUp(rotatedSize[0], 16), iDivUp(rotatedSize[1], 16), batchCount);
    sampleForward<<<grid, block>>>(workspace.current,
                                   nullptr,
                                   devVolume,
                                   devAttenuation,
                                   cudaVolumeSize,
                                   deviceMetadata.matrices,
                                   deviceMetadata.firstSlices,
                                   rotatedSize[0],
                                   rotatedSize[1],
                                   depth - 1,
                                   rotatedSpacing[2],
                                   batchStart,
                                   false);
    for (int z = depth - 2; z >= 0; --z)
    {
      gaussianBatch(workspace.current,
                    workspace.blurred,
                    workspace.scratch,
                    deviceMetadata,
                    rotatedSize[0],
                    rotatedSize[1],
                    batchStart,
                    batchCount,
                    z,
                    metadataStride);
      sampleForward<<<grid, block>>>(workspace.current,
                                     workspace.blurred,
                                     devVolume,
                                     devAttenuation,
                                     cudaVolumeSize,
                                     deviceMetadata.matrices,
                                     deviceMetadata.firstSlices,
                                     rotatedSize[0],
                                     rotatedSize[1],
                                     z,
                                     rotatedSpacing[2],
                                     batchStart,
                                     true);
    }
    gaussianBatch(workspace.current,
                  workspace.blurred,
                  workspace.scratch,
                  deviceMetadata,
                  rotatedSize[0],
                  rotatedSize[1],
                  batchStart,
                  batchCount,
                  depth,
                  metadataStride);
    finishForward<<<dim3(iDivUp(pixels, 256), batchCount), 256>>>(
      devProjectionIn, workspace.blurred, devProjectionOut, pixels, rotatedSpacing[2], batchStart);
  }
  CUDA_CHECK_ERROR;
}

void
CUDA_zeng_back_project(const int     projectionSize[3],
                       const int     volumeSize[3],
                       const int     rotatedSize[3],
                       const float   rotatedSpacing[3],
                       const float * volumeToRotatedMatrices,
                       const float * nearDistances,
                       const int *   firstSlices,
                       const float * devVolumeIn,
                       float *       devVolumeOut,
                       const float * devProjections,
                       const float * devAttenuation,
                       float         sigmaZero,
                       float         alpha,
                       unsigned int  requestedBatchSize,
                       void **       workspacePointer)
{
  if (!*workspacePointer)
    *workspacePointer = new Workspace;
  auto &           workspace = *static_cast<Workspace *>(*workspacePointer);
  const int        projections = projectionSize[2];
  const int        depth = rotatedSize[2];
  const int        pixels = rotatedSize[0] * rotatedSize[1];
  const int        metadataStride = depth + 1;
  std::vector<int> first(firstSlices, firstSlices + projections);
  const auto       key = makeMetadataKey(1,
                                   projectionSize,
                                   volumeSize,
                                   rotatedSize,
                                   rotatedSpacing,
                                   sigmaZero,
                                   alpha,
                                   volumeToRotatedMatrices,
                                   nearDistances,
                                   firstSlices);
  auto [iterator, inserted] = workspace.metadata.try_emplace(key);
  auto & deviceMetadata = iterator->second;
  if (inserted)
  {
    HostGaussianMetadata                 metadata;
    std::map<double, std::vector<float>> cache;
    std::vector<float>                   inverseMatrices(12 * projections);
    for (int projection = 0; projection < projections; ++projection)
    {
      for (int z = 0; z < depth; ++z)
      {
        float variance = 0.f;
        if (z >= firstSlices[projection] && z + 1 < depth)
        {
          const float distance = nearDistances[projection] + (z + 1 - firstSlices[projection]) * rotatedSpacing[2];
          variance = distance * 2.f * rotatedSpacing[2] * alpha * alpha + 2.f * rotatedSpacing[2] * alpha * sigmaZero -
                     alpha * alpha * rotatedSpacing[2] * rotatedSpacing[2];
        }
        appendGaussian(metadata, cache, std::max(0.f, variance), rotatedSpacing[0], rotatedSpacing[1]);
      }
      const float initialVariance =
        (alpha * nearDistances[projection] + sigmaZero) * (alpha * nearDistances[projection] + sigmaZero);
      appendGaussian(metadata, cache, initialVariance, rotatedSpacing[0], rotatedSpacing[1]);
      invertAffine(volumeToRotatedMatrices + 12 * projection, inverseMatrices.data() + 12 * projection);
    }
    uploadMetadata(deviceMetadata, std::move(metadata), volumeToRotatedMatrices, inverseMatrices, first, projections);
  }

  const unsigned int desiredBatchSize = requestedBatchSize
                                          ? std::min(requestedBatchSize, static_cast<unsigned int>(projections))
                                          : automaticBatchSize(workspace, projections, depth, pixels, true);
  const unsigned int batchSize = allocateBatch(workspace, desiredBatchSize, depth, pixels, true);
  const size_t       volumeBytes = static_cast<size_t>(volumeSize[0]) * volumeSize[1] * volumeSize[2] * sizeof(float);
  if (devVolumeOut != devVolumeIn)
    cudaMemcpy(devVolumeOut, devVolumeIn, volumeBytes, cudaMemcpyDeviceToDevice);
  cudaArray *         attenuationArray = nullptr;
  cudaTextureObject_t attenuationTexture = 0;
  if (devAttenuation)
    prepareScalarTextureObject(const_cast<int *>(volumeSize),
                               const_cast<float *>(devAttenuation),
                               attenuationArray,
                               attenuationTexture,
                               false,
                               true,
                               cudaAddressModeClamp);
  const dim3 block2(16, 16);
  const dim3 block3(8, 8, 4);
  const dim3 grid3(iDivUp(volumeSize[0], 8), iDivUp(volumeSize[1], 8), iDivUp(volumeSize[2], 4));
  const int3 cudaVolumeSize = make_int3(volumeSize[0], volumeSize[1], volumeSize[2]);
  const int3 cudaRotatedSize = make_int3(rotatedSize[0], rotatedSize[1], rotatedSize[2]);
  for (int batchStart = 0; batchStart < projections; batchStart += batchSize)
  {
    const int  batchCount = std::min<int>(batchSize, projections - batchStart);
    float *    current = workspace.current;
    float *    blurred = workspace.blurred;
    const dim3 grid2(iDivUp(rotatedSize[0], 16), iDivUp(rotatedSize[1], 16), batchCount);
    cudaMemset(workspace.rotated, 0, static_cast<size_t>(batchCount) * depth * pixels * sizeof(float));
    copyProjections<<<dim3(iDivUp(pixels, 256), batchCount), 256>>>(devProjections, current, pixels, batchStart);
    if (attenuationTexture)
      attenuate<<<grid2, block2>>>(current,
                                   attenuationTexture,
                                   cudaVolumeSize,
                                   deviceMetadata.inverseMatrices,
                                   rotatedSize[0],
                                   rotatedSize[1],
                                   0,
                                   deviceMetadata.firstSlices,
                                   true,
                                   rotatedSpacing[2],
                                   batchStart);
    gaussianBatch(current,
                  blurred,
                  workspace.scratch,
                  deviceMetadata,
                  rotatedSize[0],
                  rotatedSize[1],
                  batchStart,
                  batchCount,
                  depth,
                  metadataStride);
    std::swap(current, blurred);
    for (int z = 0; z < depth; ++z)
    {
      storeSlices<<<dim3(iDivUp(pixels, 256), batchCount), 256>>>(
        current, workspace.rotated, pixels, depth, z, deviceMetadata.firstSlices, batchStart);
      if (z + 1 == depth)
        break;
      if (attenuationTexture)
        attenuate<<<grid2, block2>>>(current,
                                     attenuationTexture,
                                     cudaVolumeSize,
                                     deviceMetadata.inverseMatrices,
                                     rotatedSize[0],
                                     rotatedSize[1],
                                     z + 1,
                                     deviceMetadata.firstSlices,
                                     false,
                                     rotatedSpacing[2],
                                     batchStart);
      gaussianBatch(current,
                    blurred,
                    workspace.scratch,
                    deviceMetadata,
                    rotatedSize[0],
                    rotatedSize[1],
                    batchStart,
                    batchCount,
                    z,
                    metadataStride);
      std::swap(current, blurred);
    }
    addRotatedBatch<<<grid3, block3>>>(workspace.rotated,
                                       devVolumeOut,
                                       cudaVolumeSize,
                                       cudaRotatedSize,
                                       deviceMetadata.matrices,
                                       batchStart,
                                       batchCount,
                                       rotatedSpacing[2]);
  }
  if (attenuationArray)
  {
    cudaDestroyTextureObject(attenuationTexture);
    cudaFreeArray(attenuationArray);
  }
  CUDA_CHECK_ERROR;
}

void
CUDA_zeng_release_workspace(void * workspace)
{
  delete static_cast<Workspace *>(workspace);
}
