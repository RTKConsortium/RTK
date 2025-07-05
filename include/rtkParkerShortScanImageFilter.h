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

#ifndef rtkParkerShortScanImageFilter_h
#define rtkParkerShortScanImageFilter_h

#include <itkInPlaceImageFilter.h>

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConfiguration.h"

namespace rtk
{

/** \class ParkerShortScanImageFilter
 *
 * Weighting of image projections to handle short-scans
 * in tomography reconstruction. Based on [Parker, Med Phys, 1982].
 * Class implements a fix to typo in equation (12) of Parker as seen
 * in book "Principles of computerized tomographic imaging" by Kak and Slaney
 *
 * \test rtkshortscantest.cxx, rtkshortscancompcudatest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK InPlaceImageFilter
 */

template <class TInputImage, class TOutputImage = TInputImage>
class ITK_TEMPLATE_EXPORT ParkerShortScanImageFilter : public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ParkerShortScanImageFilter);

  /** Standard class type alias. */
  using Self = ParkerShortScanImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
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
  itkOverrideGetNameOfClassMacro(ParkerShortScanImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetModifiableObjectMacro(Geometry, GeometryType);
  itkSetObjectMacro(Geometry, GeometryType);

  /** Get / Set the angular gap threshold above which a short scan is detected. */
  itkGetMacro(AngularGapThreshold, double);
  itkSetMacro(AngularGapThreshold, double);

protected:
  ParkerShortScanImageFilter() = default;
  ~ParkerShortScanImageFilter() override = default;

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() const override;

  void
  GenerateInputRequestedRegion() override;

  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

  /** Actual angular gap in the projections */
  double m_Delta;

  /** First angle of the short scan */
  double m_FirstAngle;

  /** Internal variable indicating whether this scan is short */
  bool m_IsShortScan{ false };

private:
  /** RTK geometry object */
  GeometryPointer m_Geometry;

  /** Superior and inferior position of the detector along the weighting direction, i.e. x.
   * The computed value account for the x projection offset of the geometry.
   */
  double m_InferiorCorner;
  double m_SuperiorCorner;

  /** Minimum angular gap to automatically detect a short scan. Defaults is pi/9 radians. */
  double m_AngularGapThreshold{ itk::Math::pi / 9 };
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkParkerShortScanImageFilter.hxx"
#endif

#endif
