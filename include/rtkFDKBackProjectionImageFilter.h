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

#ifndef rtkFDKBackProjectionImageFilter_h
#define rtkFDKBackProjectionImageFilter_h

#include "rtkBackProjectionImageFilter.h"

namespace rtk
{

/** \class FDKBackProjectionImageFilter
 * \brief CPU version of the backprojection of the FDK algorithm.
 *
 * CPU implementation of the backprojection step of the
 * [Feldkamp, Davis, Kress, 1984] algorithm for filtered backprojection
 * reconstruction of cone-beam CT images with a circular source trajectory.
 *
 * \author Simon Rit
 *
 * \ingroup RTK Projector
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT FDKBackProjectionImageFilter : public BackProjectionImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(FDKBackProjectionImageFilter);

  /** Standard class type alias. */
  using Self = FDKBackProjectionImageFilter;
  using Superclass = BackProjectionImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using ProjectionMatrixType = typename Superclass::ProjectionMatrixType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using ProjectionImageType = typename Superclass::ProjectionImageType;
  using ProjectionImagePointer = typename ProjectionImageType::Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FDKBackProjectionImageFilter, ImageToImageFilter);

protected:
  FDKBackProjectionImageFilter() = default;
  ~FDKBackProjectionImageFilter() override = default;

  void
  GenerateOutputInformation() override;

  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

  /** Optimized version when the rotation is parallel to X, i.e. matrix[1][0]
    and matrix[2][0] are zeros. */
  void
  OptimizedBackprojectionX(const OutputImageRegionType & region,
                           const ProjectionMatrixType &  matrix,
                           const ProjectionImagePointer  projection) override;

  /** Optimized version when the rotation is parallel to Y, i.e. matrix[1][1]
    and matrix[2][1] are zeros. */
  void
  OptimizedBackprojectionY(const OutputImageRegionType & region,
                           const ProjectionMatrixType &  matrix,
                           const ProjectionImagePointer  projection) override;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkFDKBackProjectionImageFilter.hxx"
#endif

#endif
