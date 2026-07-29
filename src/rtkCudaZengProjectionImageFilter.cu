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
        const int x = x0 + dx, y = y0 + dy, z = z0 + dz;
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

__global__ void
sampleForwardSlice(float *       slice,
                   const float * previous,
                   const float * volume,
                   const float * attenuation,
                   int3          volumeSize,
                   const float * matrix,
                   int           width,
                   int           height,
                   int           z,
                   float         attenuationStep,
                   bool          addPrevious)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  const float3 position = applyMatrix(matrix, x, y, z);
  float        value = trilinearZero(volume, volumeSize, position);
  if (addPrevious)
    value += previous[y * width + x];
  if (attenuation)
    value *= expf(-attenuationStep * trilinearZero(attenuation, volumeSize, position));
  slice[y * width + x] = value;
}

__global__ void
gaussianXShared(const float * input, float * output, int width, int height, const float * coefficients, int radius)
{
  extern __shared__ float tile[];
  const int               x = blockIdx.x * blockDim.x + threadIdx.x;
  const int               y = blockIdx.y * blockDim.y + threadIdx.y;
  const int               tileWidth = blockDim.x + 2 * radius;
  const int               tileElements = tileWidth * blockDim.y;
  const int               threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
  const int               threadCount = blockDim.x * blockDim.y;
  for (int index = threadIndex; index < tileElements; index += threadCount)
  {
    const int localX = index % tileWidth;
    const int localY = index / tileWidth;
    const int globalX = blockIdx.x * blockDim.x + localX - radius;
    const int globalY = blockIdx.y * blockDim.y + localY;
    tile[index] = globalX >= 0 && globalX < width && globalY < height ? input[globalY * width + globalX] : 0.f;
  }
  __syncthreads();
  if (x >= width || y >= height)
    return;
  float sum = 0.f;
  for (int offset = -radius; offset <= radius; ++offset)
    sum += coefficients[abs(offset)] * tile[threadIdx.y * tileWidth + threadIdx.x + radius + offset];
  output[y * width + x] = sum;
}

__global__ void
gaussianYShared(const float * input, float * output, int width, int height, const float * coefficients, int radius)
{
  extern __shared__ float tile[];
  const int               x = blockIdx.x * blockDim.x + threadIdx.x;
  const int               y = blockIdx.y * blockDim.y + threadIdx.y;
  const int               tileHeight = blockDim.y + 2 * radius;
  const int               tileElements = blockDim.x * tileHeight;
  const int               threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
  const int               threadCount = blockDim.x * blockDim.y;
  for (int index = threadIndex; index < tileElements; index += threadCount)
  {
    const int localX = index % blockDim.x;
    const int localY = index / blockDim.x;
    const int globalX = blockIdx.x * blockDim.x + localX;
    const int globalY = blockIdx.y * blockDim.y + localY - radius;
    tile[index] = globalX < width && globalY >= 0 && globalY < height ? input[globalY * width + globalX] : 0.f;
  }
  __syncthreads();
  if (x >= width || y >= height)
    return;
  float sum = 0.f;
  for (int offset = -radius; offset <= radius; ++offset)
    sum += coefficients[abs(offset)] * tile[(threadIdx.y + radius + offset) * blockDim.x + threadIdx.x];
  output[y * width + x] = sum;
}

std::vector<float>
discreteGaussianCoefficients(double variance)
{
  if (variance <= 1.e-12)
    return { 1.f };

  constexpr double    maximumError = 1.e-5;
  constexpr size_t    maximumKernelWidth = 32;
  const double        exponential = std::exp(-variance);
  std::vector<double> coefficients;
  coefficients.push_back(exponential * std::cyl_bessel_i(0., variance));
  double sum = coefficients[0];
  for (int order = 1; sum < 1. - maximumError; ++order)
  {
    const double coefficient = exponential * std::cyl_bessel_i(static_cast<double>(order), variance);
    coefficients.push_back(coefficient);
    sum += 2. * coefficient;
    if (coefficient <= 0. || coefficients.size() > maximumKernelWidth)
      break;
  }
  std::vector<float> normalized(coefficients.size());
  std::transform(coefficients.begin(), coefficients.end(), normalized.begin(), [sum](double value) {
    return static_cast<float>(value / sum);
  });
  return normalized;
}

constexpr size_t GaussianCoefficientStride = 33;

struct GaussianMetadata
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

struct DeviceMetadata
{
  float *          coefficientsX{};
  float *          coefficientsY{};
  float *          matrices{};
  float *          inverseMatrices{};
  std::vector<int> radiiX;
  std::vector<int> radiiY;

  ~DeviceMetadata()
  {
    cudaFree(inverseMatrices);
    cudaFree(matrices);
    cudaFree(coefficientsY);
    cudaFree(coefficientsX);
  }
};

struct CudaZengWorkspace
{
  float *                               current{};
  float *                               blurred{};
  float *                               gaussianScratch{};
  float *                               rotated{};
  size_t                                sliceCapacity{};
  size_t                                rotatedCapacity{};
  std::map<MetadataKey, DeviceMetadata> metadata;

  ~CudaZengWorkspace()
  {
    cudaFree(rotated);
    cudaFree(gaussianScratch);
    cudaFree(blurred);
    cudaFree(current);
  }
};

void
ensureSliceBuffers(CudaZengWorkspace & workspace, size_t requiredElements)
{
  if (requiredElements <= workspace.sliceCapacity)
    return;
  cudaFree(workspace.gaussianScratch);
  cudaFree(workspace.blurred);
  cudaFree(workspace.current);
  cudaMalloc(&workspace.current, requiredElements * sizeof(float));
  cudaMalloc(&workspace.blurred, requiredElements * sizeof(float));
  cudaMalloc(&workspace.gaussianScratch, requiredElements * sizeof(float));
  workspace.sliceCapacity = requiredElements;
}

void
ensureRotatedBuffer(CudaZengWorkspace & workspace, size_t requiredElements)
{
  if (requiredElements <= workspace.rotatedCapacity)
    return;
  cudaFree(workspace.rotated);
  cudaMalloc(&workspace.rotated, requiredElements * sizeof(float));
  workspace.rotatedCapacity = requiredElements;
}

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
  if (firstSlices)
    key.integers.insert(key.integers.end(), firstSlices, firstSlices + projectionSize[2]);
  key.values.insert(key.values.end(), rotatedSpacing, rotatedSpacing + 3);
  key.values.push_back(sigmaZero);
  key.values.push_back(alpha);
  key.values.insert(key.values.end(), matrices, matrices + 12 * projectionSize[2]);
  key.values.insert(key.values.end(), distances, distances + projectionSize[2]);
  return key;
}

void
appendGaussianMetadata(GaussianMetadata &                     metadata,
                       std::map<double, std::vector<float>> & coefficientCache,
                       float                                  variance,
                       float                                  spacingX,
                       float                                  spacingY)
{
  const double         variances[] = { variance / (spacingX * spacingX), variance / (spacingY * spacingY) };
  std::vector<float> * packed[] = { &metadata.coefficientsX, &metadata.coefficientsY };
  std::vector<int> *   radii[] = { &metadata.radiiX, &metadata.radiiY };
  for (int dimension = 0; dimension < 2; ++dimension)
  {
    auto [iterator, inserted] = coefficientCache.try_emplace(variances[dimension]);
    if (inserted)
      iterator->second = discreteGaussianCoefficients(variances[dimension]);
    const auto & coefficients = iterator->second;
    radii[dimension]->push_back(static_cast<int>(coefficients.size()) - 1);
    packed[dimension]->insert(packed[dimension]->end(), coefficients.begin(), coefficients.end());
    packed[dimension]->resize(packed[dimension]->size() + GaussianCoefficientStride - coefficients.size(), 0.f);
  }
}

void
gaussian2D(const float * input,
           float *       output,
           float *       scratch,
           const float * deviceCoefficientsX,
           const float * deviceCoefficientsY,
           int           radiusX,
           int           radiusY,
           int           width,
           int           height,
           dim3          grid,
           dim3          block)
{
  const size_t sharedX = (block.x + 2 * radiusX) * block.y * sizeof(float);
  const size_t sharedY = block.x * (block.y + 2 * radiusY) * sizeof(float);
  gaussianXShared<<<grid, block, sharedX>>>(input, scratch, width, height, deviceCoefficientsX, radiusX);
  gaussianYShared<<<grid, block, sharedY>>>(scratch, output, width, height, deviceCoefficientsY, radiusY);
}

__global__ void
finishForward(const float * input, const float * zeng, float * output, int count, float thickness)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    output[i] = input[i] + thickness * zeng[i];
}

__global__ void
copyProjection(const float * projections, float * slice, int count, int projection)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    slice[i] = projections[projection * count + i];
}

__global__ void
attenuateSlice(float *             slice,
               cudaTextureObject_t attenuation,
               const float *       rotatedToVolume,
               int                 width,
               int                 height,
               int                 z,
               float               attenuationStep)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  const float3 position = applyMatrix(rotatedToVolume, x, y, z);
  slice[y * width + x] *= expf(-attenuationStep * tex3D<float>(attenuation, position.x, position.y, position.z));
}

__global__ void
storeSlice(const float * slice, float * volume, int count, int z)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    volume[z * count + i] = slice[i];
}

__global__ void
addRotatedVolume(const float * rotated,
                 const float * input,
                 float *       output,
                 int3          volumeSize,
                 int3          rotatedSize,
                 const float * volumeToRotated,
                 float         thickness)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= volumeSize.x || y >= volumeSize.y || z >= volumeSize.z)
    return;
  const int    index = (z * volumeSize.y + y) * volumeSize.x + x;
  const float3 position = applyMatrix(volumeToRotated, x, y, z);
  output[index] = input[index] + thickness * trilinearZero(rotated, rotatedSize, position);
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
                          void **       workspacePointer)
{
  const int3 cudaVolumeSize = make_int3(volumeSize[0], volumeSize[1], volumeSize[2]);
  if (!*workspacePointer)
    *workspacePointer = new CudaZengWorkspace;
  auto &    workspace = *static_cast<CudaZengWorkspace *>(*workspacePointer);
  const int pixelCount = rotatedSize[0] * rotatedSize[1];
  ensureSliceBuffers(workspace, pixelCount);

  const auto key = makeMetadataKey(0,
                                   projectionSize,
                                   volumeSize,
                                   rotatedSize,
                                   rotatedSpacing,
                                   sigmaZero,
                                   alpha,
                                   rotatedToVolumeMatrices,
                                   farDistances,
                                   nullptr);
  auto [metadataIterator, inserted] = workspace.metadata.try_emplace(key);
  auto & deviceMetadata = metadataIterator->second;
  if (inserted)
  {
    GaussianMetadata                     metadata;
    std::map<double, std::vector<float>> coefficientCache;
    for (int projection = 0; projection < projectionSize[2]; ++projection)
    {
      float distance = farDistances[projection];
      for (int z = rotatedSize[2] - 2; z >= 0 && distance - rotatedSpacing[2] >= 0.f; --z)
      {
        const float variance = distance * 2.f * rotatedSpacing[2] * alpha * alpha +
                               2.f * rotatedSpacing[2] * alpha * sigmaZero -
                               alpha * alpha * rotatedSpacing[2] * rotatedSpacing[2];
        appendGaussianMetadata(
          metadata, coefficientCache, std::max(0.f, variance), rotatedSpacing[0], rotatedSpacing[1]);
        distance -= rotatedSpacing[2];
      }
      const float finalVariance = (alpha * distance + sigmaZero) * (alpha * distance + sigmaZero);
      appendGaussianMetadata(metadata, coefficientCache, finalVariance, rotatedSpacing[0], rotatedSpacing[1]);
    }
    deviceMetadata.radiiX = std::move(metadata.radiiX);
    deviceMetadata.radiiY = std::move(metadata.radiiY);
    cudaMalloc(&deviceMetadata.coefficientsX, metadata.coefficientsX.size() * sizeof(float));
    cudaMalloc(&deviceMetadata.coefficientsY, metadata.coefficientsY.size() * sizeof(float));
    cudaMalloc(&deviceMetadata.matrices, 12 * projectionSize[2] * sizeof(float));
    cudaMemcpy(deviceMetadata.coefficientsX,
               metadata.coefficientsX.data(),
               metadata.coefficientsX.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMetadata.coefficientsY,
               metadata.coefficientsY.data(),
               metadata.coefficientsY.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(
      deviceMetadata.matrices, rotatedToVolumeMatrices, 12 * projectionSize[2] * sizeof(float), cudaMemcpyHostToDevice);
  }

  float *    current = workspace.current;
  float *    blurred = workspace.blurred;
  float *    gaussianScratch = workspace.gaussianScratch;
  const dim3 block(16, 16);
  const dim3 grid(iDivUp(rotatedSize[0], 16), iDivUp(rotatedSize[1], 16));

  size_t gaussianIndex = 0;
  for (int projection = 0; projection < projectionSize[2]; ++projection)
  {
    const float * matrix = deviceMetadata.matrices + 12 * projection;
    float         distance = farDistances[projection];
    int           z = rotatedSize[2] - 1;
    sampleForwardSlice<<<grid, block>>>(current,
                                        nullptr,
                                        devVolume,
                                        devAttenuation,
                                        cudaVolumeSize,
                                        matrix,
                                        rotatedSize[0],
                                        rotatedSize[1],
                                        z,
                                        rotatedSpacing[2],
                                        false);
    for (--z; z >= 0 && distance - rotatedSpacing[2] >= 0.f; --z)
    {
      gaussian2D(current,
                 blurred,
                 gaussianScratch,
                 deviceMetadata.coefficientsX + GaussianCoefficientStride * gaussianIndex,
                 deviceMetadata.coefficientsY + GaussianCoefficientStride * gaussianIndex,
                 deviceMetadata.radiiX[gaussianIndex],
                 deviceMetadata.radiiY[gaussianIndex],
                 rotatedSize[0],
                 rotatedSize[1],
                 grid,
                 block);
      ++gaussianIndex;
      sampleForwardSlice<<<grid, block>>>(current,
                                          blurred,
                                          devVolume,
                                          devAttenuation,
                                          cudaVolumeSize,
                                          matrix,
                                          rotatedSize[0],
                                          rotatedSize[1],
                                          z,
                                          rotatedSpacing[2],
                                          true);
      distance -= rotatedSpacing[2];
    }
    gaussian2D(current,
               blurred,
               gaussianScratch,
               deviceMetadata.coefficientsX + GaussianCoefficientStride * gaussianIndex,
               deviceMetadata.coefficientsY + GaussianCoefficientStride * gaussianIndex,
               deviceMetadata.radiiX[gaussianIndex],
               deviceMetadata.radiiY[gaussianIndex],
               rotatedSize[0],
               rotatedSize[1],
               grid,
               block);
    ++gaussianIndex;
    finishForward<<<iDivUp(pixelCount, 256), 256>>>(devProjectionIn + projection * pixelCount,
                                                    blurred,
                                                    devProjectionOut + projection * pixelCount,
                                                    pixelCount,
                                                    rotatedSpacing[2]);
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
                       void **       workspacePointer)
{
  const size_t volumeBytes = static_cast<size_t>(volumeSize[0]) * volumeSize[1] * volumeSize[2] * sizeof(float);
  if (!*workspacePointer)
    *workspacePointer = new CudaZengWorkspace;
  auto &       workspace = *static_cast<CudaZengWorkspace *>(*workspacePointer);
  const int    slicePixels = rotatedSize[0] * rotatedSize[1];
  const size_t rotatedElements = static_cast<size_t>(slicePixels) * rotatedSize[2];
  const size_t rotatedBytes = rotatedElements * sizeof(float);
  ensureSliceBuffers(workspace, slicePixels);
  ensureRotatedBuffer(workspace, rotatedElements);

  const auto key = makeMetadataKey(1,
                                   projectionSize,
                                   volumeSize,
                                   rotatedSize,
                                   rotatedSpacing,
                                   sigmaZero,
                                   alpha,
                                   volumeToRotatedMatrices,
                                   nearDistances,
                                   firstSlices);
  auto [metadataIterator, inserted] = workspace.metadata.try_emplace(key);
  auto & deviceMetadata = metadataIterator->second;
  if (inserted)
  {
    GaussianMetadata                     metadata;
    std::map<double, std::vector<float>> coefficientCache;
    std::vector<float>                   inverseMatrices(12 * projectionSize[2]);
    for (int projection = 0; projection < projectionSize[2]; ++projection)
    {
      float distance = nearDistances[projection];
      float variance = (alpha * distance + sigmaZero) * (alpha * distance + sigmaZero);
      appendGaussianMetadata(metadata, coefficientCache, variance, rotatedSpacing[0], rotatedSpacing[1]);
      for (int z = firstSlices[projection]; z + 1 < rotatedSize[2]; ++z)
      {
        distance += rotatedSpacing[2];
        variance = distance * 2.f * rotatedSpacing[2] * alpha * alpha + 2.f * rotatedSpacing[2] * alpha * sigmaZero -
                   alpha * alpha * rotatedSpacing[2] * rotatedSpacing[2];
        appendGaussianMetadata(
          metadata, coefficientCache, std::max(0.f, variance), rotatedSpacing[0], rotatedSpacing[1]);
      }
      invertAffine(volumeToRotatedMatrices + 12 * projection, inverseMatrices.data() + 12 * projection);
    }
    deviceMetadata.radiiX = std::move(metadata.radiiX);
    deviceMetadata.radiiY = std::move(metadata.radiiY);
    cudaMalloc(&deviceMetadata.coefficientsX, metadata.coefficientsX.size() * sizeof(float));
    cudaMalloc(&deviceMetadata.coefficientsY, metadata.coefficientsY.size() * sizeof(float));
    cudaMalloc(&deviceMetadata.matrices, 12 * projectionSize[2] * sizeof(float));
    cudaMalloc(&deviceMetadata.inverseMatrices, 12 * projectionSize[2] * sizeof(float));
    cudaMemcpy(deviceMetadata.coefficientsX,
               metadata.coefficientsX.data(),
               metadata.coefficientsX.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMetadata.coefficientsY,
               metadata.coefficientsY.data(),
               metadata.coefficientsY.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(
      deviceMetadata.matrices, volumeToRotatedMatrices, 12 * projectionSize[2] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMetadata.inverseMatrices,
               inverseMatrices.data(),
               inverseMatrices.size() * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  float * rotated = workspace.rotated;
  float * current = workspace.current;
  float * blurred = workspace.blurred;
  float * gaussianScratch = workspace.gaussianScratch;
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
                               cudaAddressModeBorder);

  const dim3 block2(16, 16);
  const dim3 grid2(iDivUp(rotatedSize[0], 16), iDivUp(rotatedSize[1], 16));
  const dim3 block3(8, 8, 4);
  const dim3 grid3(iDivUp(volumeSize[0], 8), iDivUp(volumeSize[1], 8), iDivUp(volumeSize[2], 4));
  const int3 cudaVolumeSize = make_int3(volumeSize[0], volumeSize[1], volumeSize[2]);
  const int3 cudaRotatedSize = make_int3(rotatedSize[0], rotatedSize[1], rotatedSize[2]);

  size_t gaussianIndex = 0;
  for (int projection = 0; projection < projectionSize[2]; ++projection)
  {
    const float * matrix = deviceMetadata.matrices + 12 * projection;
    const float * inverseMatrix = deviceMetadata.inverseMatrices + 12 * projection;
    cudaMemset(rotated, 0, rotatedBytes);
    copyProjection<<<iDivUp(slicePixels, 256), 256>>>(devProjections, current, slicePixels, projection);
    float distance = nearDistances[projection];
    gaussian2D(current,
               blurred,
               gaussianScratch,
               deviceMetadata.coefficientsX + GaussianCoefficientStride * gaussianIndex,
               deviceMetadata.coefficientsY + GaussianCoefficientStride * gaussianIndex,
               deviceMetadata.radiiX[gaussianIndex],
               deviceMetadata.radiiY[gaussianIndex],
               rotatedSize[0],
               rotatedSize[1],
               grid2,
               block2);
    ++gaussianIndex;
    std::swap(current, blurred);

    for (int z = firstSlices[projection]; z < rotatedSize[2]; ++z)
    {
      storeSlice<<<iDivUp(slicePixels, 256), 256>>>(current, rotated, slicePixels, z);
      if (z + 1 == rotatedSize[2])
        break;
      if (attenuationTexture)
        attenuateSlice<<<grid2, block2>>>(
          current, attenuationTexture, inverseMatrix, rotatedSize[0], rotatedSize[1], z + 1, rotatedSpacing[2]);
      distance += rotatedSpacing[2];
      gaussian2D(current,
                 blurred,
                 gaussianScratch,
                 deviceMetadata.coefficientsX + GaussianCoefficientStride * gaussianIndex,
                 deviceMetadata.coefficientsY + GaussianCoefficientStride * gaussianIndex,
                 deviceMetadata.radiiX[gaussianIndex],
                 deviceMetadata.radiiY[gaussianIndex],
                 rotatedSize[0],
                 rotatedSize[1],
                 grid2,
                 block2);
      ++gaussianIndex;
      std::swap(current, blurred);
    }
    addRotatedVolume<<<grid3, block3>>>(
      rotated, devVolumeOut, devVolumeOut, cudaVolumeSize, cudaRotatedSize, matrix, rotatedSpacing[2]);
  }
  CUDA_CHECK_ERROR;
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
  delete static_cast<CudaZengWorkspace *>(workspace);
}
