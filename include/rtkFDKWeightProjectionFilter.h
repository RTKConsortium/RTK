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

#ifndef rtkFDKWeightProjectionFilter_h
#define rtkFDKWeightProjectionFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConfiguration.h"

namespace rtk
{

/** \class FDKWeightProjectionFilter
 * \brief Weighting of projections to correct for the divergence in
 * filtered backprojection reconstruction algorithms.
 * The weighting comprises:
 * - the 2D weighting of the FDK algorithm [Feldkamp, 1984],
 * - its modification described in [Rit and Clackdoyle, CT meeting, 2014] for
 *   tilted detector
 * - the correction of the ramp factor for divergent full scan,
 * - the angular weighting for the final 3D integral of FDK.
 * Note that SourceToDetectorDistance, SourceToDetectorIsocenter
 * SouceOffsets and ProjectionOffsets are accounted for on a per
 * projection basis but InPlaneRotation and OutOfPlaneRotation are not
 * accounted for.
 * \author Simon Rit
 *
 * \test rtkfdktest.cxx, rtkrampfiltertest.cxx, rtkdisplaceddetectortest.cxx,
 * rtkshortscantest.cxx, rtkfdkprojweightcompcudatest.cxx
 *
 * \ingroup RTK InPlaceImageFilter
 */

template <class TInputImage, class TOutputImage = TInputImage>
class ITK_TEMPLATE_EXPORT FDKWeightProjectionFilter : public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(FDKWeightProjectionFilter);

  /** Standard class type alias. */
  using Self = FDKWeightProjectionFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using OutputImageRegionType = typename OutputImageType::RegionType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
#ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(FDKWeightProjectionFilter);
#else
  itkTypeMacro(FDKWeightProjectionFilter, itk::ImageToImageFilter);
#endif

  /** Get/ Set geometry structure */
  itkGetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);
  itkSetObjectMacro(Geometry, ThreeDCircularProjectionGeometry);

protected:
  FDKWeightProjectionFilter() = default;
  ~FDKWeightProjectionFilter() override = default;

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() ITKv5_CONST override;

  void
  BeforeThreadedGenerateData() override;

  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

private:
  /** Angular weights for each projection */
  std::vector<double> m_ConstantProjectionFactor;

  /** Tilt angles with respect to the conventional situation */
  std::vector<double> m_TiltAngles;

  /** Geometrical description of the system */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkFDKWeightProjectionFilter.hxx"
#endif

#endif
