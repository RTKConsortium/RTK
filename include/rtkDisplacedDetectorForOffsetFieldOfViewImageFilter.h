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

#ifndef rtkDisplacedDetectorForOffsetFieldOfViewImageFilter_h
#define rtkDisplacedDetectorForOffsetFieldOfViewImageFilter_h

#include "rtkDisplacedDetectorImageFilter.h"

namespace rtk
{

/** \class DisplacedDetectorForOffsetFieldOfViewImageFilter
 * \brief Weigting for displaced detectors with offset field-of-view
 *
 * The class does something similar to rtk::DisplacedDetectorImageFilter but
 * handles in addition the case of a field-of-view that is not centered on the
 * center of rotation.
 *
 * \test rtkdisplaceddetectorcompoffsettest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK ImageToImageFilter
 */
template <class TInputImage, class TOutputImage = TInputImage>
class ITK_TEMPLATE_EXPORT DisplacedDetectorForOffsetFieldOfViewImageFilter
  : public rtk::DisplacedDetectorImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DisplacedDetectorForOffsetFieldOfViewImageFilter);

  /** Standard class type alias. */
  using Self = DisplacedDetectorForOffsetFieldOfViewImageFilter;
  using Superclass = rtk::DisplacedDetectorImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using OutputImageRegionType = typename OutputImageType::RegionType;
  using WeightImageType = itk::Image<typename TOutputImage::PixelType, 1>;

  using GeometryType = ThreeDCircularProjectionGeometry;
  using GeometryPointer = GeometryType::Pointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(DisplacedDetectorForOffsetFieldOfViewImageFilter);

protected:
  DisplacedDetectorForOffsetFieldOfViewImageFilter() = default;
  ~DisplacedDetectorForOffsetFieldOfViewImageFilter() override = default;

  void
  GenerateOutputInformation() override;

  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

private:
  /**
   * Center coordinates and size of the FOV cylinder.
   */
  double m_FOVRadius{ -1. };
  double m_FOVCenterX{ 0. };
  double m_FOVCenterZ{ 0. };
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkDisplacedDetectorForOffsetFieldOfViewImageFilter.hxx"
#endif

#endif
