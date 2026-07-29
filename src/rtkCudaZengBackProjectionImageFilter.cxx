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
#include "rtkCudaZengBackProjectionImageFilter.h"

#ifdef RTK_USE_CUDA
#  include "rtkCudaZengProjectionImageFilter.hcu"
#  include "rtkHomogeneousMatrix.h"
#  include <itkCenteredEuler3DTransform.h>

namespace rtk
{
namespace
{
using MatrixType = ThreeDCircularProjectionGeometry::ThreeDHomogeneousMatrixType;

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
StoreMatrix(const MatrixType & matrix, float * destination)
{
  for (unsigned int r = 0; r < 3; ++r)
    for (unsigned int c = 0; c < 4; ++c)
      destination[4 * r + c] = static_cast<float>(matrix[r][c]);
}
} // namespace

CudaZengBackProjectionImageFilter::CudaZengBackProjectionImageFilter() { this->InPlaceOff(); }

CudaZengBackProjectionImageFilter::~CudaZengBackProjectionImageFilter()
{
  CUDA_zeng_release_workspace(m_CudaWorkspace);
}

void
CudaZengBackProjectionImageFilter::GPUGenerateData()
{
  const auto * geometry = this->GetGeometry();
  if (!geometry)
    itkGenericExceptionMacro(<< "CudaZengBackProjectionImageFilter requires a projection geometry.");

  const auto &       volumeRegion = this->GetOutput()->GetBufferedRegion();
  const auto &       projectionRegion = this->GetInput(1)->GetBufferedRegion();
  const unsigned int firstProjection = this->GetInput(1)->GetRequestedRegion().GetIndex(2);
  const unsigned int numberOfProjections = this->GetInput(1)->GetRequestedRegion().GetSize(2);
  int                projectionSize[3] = { static_cast<int>(projectionRegion.GetSize(0)),
                                           static_cast<int>(projectionRegion.GetSize(1)),
                                           static_cast<int>(numberOfProjections) };
  int                volumeSize[3] = { static_cast<int>(volumeRegion.GetSize(0)),
                                       static_cast<int>(volumeRegion.GetSize(1)),
                                       static_cast<int>(volumeRegion.GetSize(2)) };
  int                rotatedSize[3] = { projectionSize[0],
                                        projectionSize[1],
                                        static_cast<int>(std::ceil(volumeRegion.GetSize(2) * std::sqrt(2.0))) };
  float              rotatedSpacing[3] = { static_cast<float>(this->GetInput(1)->GetSpacing()[0]),
                                           static_cast<float>(this->GetInput(1)->GetSpacing()[1]),
                                           static_cast<float>(this->GetInput(0)->GetSpacing()[2]) };

  itk::ContinuousIndex<double, 3> centerIndex;
  for (unsigned int d = 0; d < 3; ++d)
    centerIndex[d] = volumeRegion.GetIndex(d) + (volumeRegion.GetSize(d) - 1.0) / 2.0;
  CPUImageType::PointType volumeCenter;
  this->GetInput(0)->TransformContinuousIndexToPhysicalPoint(centerIndex, volumeCenter);

  auto                      rotatedImage = CPUImageType::New();
  CPUImageType::RegionType  rotatedRegion;
  CPUImageType::IndexType   zeroIndex{};
  CPUImageType::SizeType    rotatedImageSize;
  CPUImageType::SpacingType spacing;
  for (unsigned int d = 0; d < 3; ++d)
  {
    rotatedImageSize[d] = rotatedSize[d];
    spacing[d] = rotatedSpacing[d];
  }
  rotatedRegion.SetIndex(zeroIndex);
  rotatedRegion.SetSize(rotatedImageSize);
  rotatedImage->SetRegions(rotatedRegion);
  rotatedImage->SetSpacing(spacing);
  rotatedImage->SetDirection(this->GetInput(1)->GetDirection());

  std::vector<float> matrices(12 * numberOfProjections);
  std::vector<float> nearDistances(numberOfProjections);
  std::vector<int>   firstSlices(numberOfProjections);
  MatrixType         volumeIndexTranslation;
  volumeIndexTranslation.SetIdentity();
  for (unsigned int d = 0; d < 3; ++d)
    volumeIndexTranslation[d][3] = volumeRegion.GetIndex(d);
  using TransformType = itk::CenteredEuler3DTransform<double>;
  auto                          transform = TransformType::New();
  TransformType::InputPointType zero{};
  transform->SetCenter(zero);

  for (unsigned int local = 0; local < numberOfProjections; ++local)
  {
    const unsigned int projection = firstProjection + local;
    const double       angle = geometry->GetGantryAngles()[projection];
    transform->SetRotation(0., angle, 0.);
    transform->SetTranslation(itk::MakeVector(geometry->GetProjectionOffsetsX()[projection] * std::cos(-angle),
                                              geometry->GetProjectionOffsetsY()[projection],
                                              geometry->GetProjectionOffsetsX()[projection] * std::sin(-angle)));
    const auto              rotatedCenter = transform->GetMatrix() * volumeCenter;
    CPUImageType::PointType origin;
    origin[0] = this->GetInput(1)->GetOrigin()[0];
    origin[1] = this->GetInput(1)->GetOrigin()[1];
    origin[2] = rotatedCenter[2] - spacing[2] * (rotatedSize[2] - 1.0) / 2.0;
    rotatedImage->SetOrigin(origin);

    auto inverse = TransformType::New();
    if (!transform->GetInverse(inverse))
      itkGenericExceptionMacro(<< "Could not invert Zeng rotation transform.");
    const MatrixType matrix = GetPhysicalPointToIndexMatrix(rotatedImage.GetPointer()).GetVnlMatrix() *
                              TransformMatrix(inverse).GetVnlMatrix() *
                              GetIndexToPhysicalPointMatrix(this->GetInput(0)).GetVnlMatrix() *
                              volumeIndexTranslation.GetVnlMatrix();
    StoreMatrix(matrix, matrices.data() + 12 * local);

    const double firstDistance = geometry->GetSourceToIsocenterDistances()[projection] + origin[2];
    firstSlices[local] = std::max(0, static_cast<int>(std::ceil(-firstDistance / spacing[2])));
    nearDistances[local] = static_cast<float>(firstDistance + firstSlices[local] * spacing[2]);
  }

  const auto    projectionOffset = firstProjection - projectionRegion.GetIndex(2);
  const auto    pixelsPerProjection = projectionSize[0] * projectionSize[1];
  const float * projections = static_cast<float *>(this->GetInput(1)->GetCudaDataManager()->GetGPUBufferPointer()) +
                              projectionOffset * pixelsPerProjection;
  const float * volumeIn = static_cast<float *>(this->GetInput(0)->GetCudaDataManager()->GetGPUBufferPointer());
  float *       volumeOut = static_cast<float *>(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());
  const float * attenuation =
    this->GetInput(2) ? static_cast<float *>(this->GetInput(2)->GetCudaDataManager()->GetGPUBufferPointer()) : nullptr;
  CUDA_zeng_back_project(projectionSize,
                         volumeSize,
                         rotatedSize,
                         rotatedSpacing,
                         matrices.data(),
                         nearDistances.data(),
                         firstSlices.data(),
                         volumeIn,
                         volumeOut,
                         projections,
                         attenuation,
                         static_cast<float>(m_SigmaZero),
                         static_cast<float>(m_Alpha),
                         &m_CudaWorkspace);
}

} // namespace rtk
#endif
