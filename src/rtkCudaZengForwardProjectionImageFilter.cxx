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
#include "rtkCudaZengForwardProjectionImageFilter.h"

#ifdef RTK_USE_CUDA
#  include "rtkCudaZengProjectionImageFilter.hcu"
#  include "rtkHomogeneousMatrix.h"
#  include <itkCenteredEuler3DTransform.h>

namespace rtk
{
namespace
{
using MatrixType = ThreeDCircularProjectionGeometry::ThreeDHomogeneousMatrixType;

float *
GetCudaBufferPointer(const itk::CudaDataManager::Pointer & manager)
{
  void *                candidate = manager->GetGPUBufferPointer();
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, candidate) == cudaSuccess &&
      (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged))
    return static_cast<float *>(candidate);

  // CudaCommon versions before 2.0 returned void** here. Current versions
  // return void*. Supporting both keeps the filter GPU-resident with either API.
  cudaGetLastError();
  void * indirectCandidate = *static_cast<void **>(candidate);
  if (cudaPointerGetAttributes(&attributes, indirectCandidate) == cudaSuccess &&
      (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged))
    return static_cast<float *>(indirectCandidate);

  cudaGetLastError();
  itkGenericExceptionMacro(<< "CudaDataManager did not provide a valid CUDA buffer.");
}

MatrixType
TransformMatrix(const itk::CenteredEuler3DTransform<double> * transform)
{
  MatrixType matrix;
  matrix.SetIdentity();
  for (unsigned int r = 0; r < 3; ++r)
  {
    for (unsigned int c = 0; c < 3; ++c)
      matrix[r][c] = transform->GetMatrix()[r][c];
    matrix[r][3] = transform->GetOffset()[r];
  }
  return matrix;
}

void
StoreTextureMatrix(const MatrixType & matrix, const itk::Index<3> & bufferedIndex, float * destination)
{
  for (unsigned int r = 0; r < 3; ++r)
    for (unsigned int c = 0; c < 4; ++c)
      destination[4 * r + c] = static_cast<float>(matrix[r][c]);
  destination[3] -= bufferedIndex[0];
  destination[7] -= bufferedIndex[1];
  destination[11] -= bufferedIndex[2];
}
} // namespace

CudaZengForwardProjectionImageFilter::CudaZengForwardProjectionImageFilter() { this->InPlaceOff(); }

CudaZengForwardProjectionImageFilter::~CudaZengForwardProjectionImageFilter()
{
  CUDA_zeng_release_workspace(m_CudaWorkspace);
}

void
CudaZengForwardProjectionImageFilter::GPUGenerateData()
{
  const auto * geometry = this->GetGeometry();
  if (!geometry)
    itkGenericExceptionMacro(<< "CudaZengForwardProjectionImageFilter requires a projection geometry.");

  constexpr unsigned int Dimension = 3;
  const auto &           projectionRegion = this->GetOutput()->GetBufferedRegion();
  const auto &           volumeRegion = this->GetInput(1)->GetBufferedRegion();
  const unsigned int     firstProjection = this->GetOutput()->GetRequestedRegion().GetIndex(2);
  const unsigned int     numberOfProjections = this->GetOutput()->GetRequestedRegion().GetSize(2);

  int   projectionSize[3] = { static_cast<int>(projectionRegion.GetSize(0)),
                              static_cast<int>(projectionRegion.GetSize(1)),
                              static_cast<int>(numberOfProjections) };
  int   volumeSize[3] = { static_cast<int>(volumeRegion.GetSize(0)),
                          static_cast<int>(volumeRegion.GetSize(1)),
                          static_cast<int>(volumeRegion.GetSize(2)) };
  int   rotatedSize[3] = { projectionSize[0],
                           projectionSize[1],
                           static_cast<int>(std::ceil(volumeRegion.GetSize(2) * std::sqrt(2.0))) };
  float rotatedSpacing[3] = { static_cast<float>(this->GetInput(0)->GetSpacing()[0]),
                              static_cast<float>(this->GetInput(0)->GetSpacing()[1]),
                              static_cast<float>(this->GetInput(1)->GetSpacing()[2]) };

  CPUImageType::PointType                 volumeCenter;
  itk::ContinuousIndex<double, Dimension> centerIndex;
  for (unsigned int d = 0; d < Dimension; ++d)
    centerIndex[d] = volumeRegion.GetIndex(d) + (volumeRegion.GetSize(d) - 1.0) / 2.0;
  this->GetInput(1)->TransformContinuousIndexToPhysicalPoint(centerIndex, volumeCenter);

  std::vector<float> matrices(12 * numberOfProjections);
  std::vector<float> farDistances(numberOfProjections);
  using TransformType = itk::CenteredEuler3DTransform<double>;
  auto                          transform = TransformType::New();
  TransformType::InputPointType zero{};
  transform->SetCenter(zero);

  auto                        rotatedImage = CPUImageType::New();
  CPUImageType::SpacingType   spacing;
  CPUImageType::PointType     origin;
  CPUImageType::DirectionType direction = this->GetInput(0)->GetDirection();
  CPUImageType::RegionType    rotatedRegion;
  CPUImageType::IndexType     rotatedIndex{};
  CPUImageType::SizeType      rotatedImageSize;
  for (unsigned int d = 0; d < 3; ++d)
  {
    spacing[d] = rotatedSpacing[d];
    rotatedImageSize[d] = rotatedSize[d];
  }
  rotatedRegion.SetIndex(rotatedIndex);
  rotatedRegion.SetSize(rotatedImageSize);
  rotatedImage->SetRegions(rotatedRegion);
  rotatedImage->SetSpacing(spacing);
  rotatedImage->SetDirection(direction);

  for (unsigned int local = 0; local < numberOfProjections; ++local)
  {
    const unsigned int projection = firstProjection + local;
    const double       angle = geometry->GetGantryAngles()[projection];
    transform->SetRotation(0., angle, 0.);
    transform->SetTranslation(itk::MakeVector(geometry->GetProjectionOffsetsX()[projection] * std::cos(-angle),
                                              geometry->GetProjectionOffsetsY()[projection],
                                              geometry->GetProjectionOffsetsX()[projection] * std::sin(-angle)));

    const auto rotatedCenter = transform->GetMatrix() * volumeCenter;
    origin[0] = this->GetInput(0)->GetOrigin()[0];
    origin[1] = this->GetInput(0)->GetOrigin()[1];
    origin[2] = rotatedCenter[2] - spacing[2] * (rotatedSize[2] - 1.0) / 2.0;
    rotatedImage->SetOrigin(origin);

    const MatrixType matrix = GetPhysicalPointToIndexMatrix(this->GetInput(1)).GetVnlMatrix() *
                              TransformMatrix(transform).GetVnlMatrix() *
                              GetIndexToPhysicalPointMatrix(rotatedImage.GetPointer()).GetVnlMatrix();
    StoreTextureMatrix(matrix, volumeRegion.GetIndex(), matrices.data() + 12 * local);
    farDistances[local] = static_cast<float>(geometry->GetSourceToIsocenterDistances()[projection] + origin[2] +
                                             spacing[2] * (rotatedSize[2] - 1));
  }

  const auto    projectionOffset = firstProjection - projectionRegion.GetIndex(2);
  const auto    pixelsPerProjection = projectionSize[0] * projectionSize[1];
  const float * projectionIn =
    GetCudaBufferPointer(this->GetInput(0)->GetCudaDataManager()) + projectionOffset * pixelsPerProjection;
  float * projectionOut =
    GetCudaBufferPointer(this->GetOutput()->GetCudaDataManager()) + projectionOffset * pixelsPerProjection;
  const float * volume = GetCudaBufferPointer(this->GetInput(1)->GetCudaDataManager());
  const float * attenuation =
    this->GetInput(2) ? GetCudaBufferPointer(this->GetInput(2)->GetCudaDataManager()) : nullptr;

  CUDA_zeng_forward_project(projectionSize,
                            volumeSize,
                            rotatedSize,
                            rotatedSpacing,
                            matrices.data(),
                            farDistances.data(),
                            projectionIn,
                            projectionOut,
                            volume,
                            attenuation,
                            static_cast<float>(m_SigmaZero),
                            static_cast<float>(m_Alpha),
                            &m_CudaWorkspace);
}

} // namespace rtk
#endif
