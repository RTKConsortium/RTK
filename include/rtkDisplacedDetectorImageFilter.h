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

#ifndef rtkDisplacedDetectorImageFilter_h
#define rtkDisplacedDetectorImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConfiguration.h"

namespace rtk
{

/** \class DisplacedDetectorImageFilter
 * \brief Weigting for displaced detectors
 *
 * Weighting of image projections to handle off-centered panels
 * in tomography reconstruction. Based on [Wang, Med Phys, 2002].
 *
 * Note that the filter does nothing if the panel shift is less than 10%
 * of its size. Otherwise, it does the weighting described in the publication
 * and zero pads the data on the nearest side to the center.
 * Therefore, the InPlace capability depends on the displacement.
 * It can only be inplace if there is no displacement, it can not otherwise.
 * The GenerateOutputInformation method takes care of properly setting this up.
 *
 * By default, it computes the minimum and maximum offsets from the complete geometry object.
 * When an independent projection has to be processed, these values have to be set by the user from
 * a priori knowledge of the detector displacements.
 *
 * The weighting accounts for variations in SourceToDetectorDistances,
 * SourceOffsetsX and ProjectionOffsetsX. It currently assumes constant
 * SourceToIsocenterDistances and 0. InPlaneAngles. The other parameters are
 * not relevant in the computation because the weighting is reproduced at
 * every gantry angle on each line of the projection images.
 *
 * \test rtkdisplaceddetectortest.cxx, rtkdisplaceddetectorcompcudatest.cxx,
 * rtkdisplaceddetectorcompoffsettest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK ImageToImageFilter
 */
template <class TInputImage, class TOutputImage = TInputImage>
class ITK_TEMPLATE_EXPORT DisplacedDetectorImageFilter : public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DisplacedDetectorImageFilter);

  /** Standard class type alias. */
  using Self = DisplacedDetectorImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  static constexpr unsigned int NDimension = TInputImage::ImageDimension;
  using OutputImageRegionType = typename OutputImageType::RegionType;
  using WeightImageType = itk::Image<double, 1>;

  using GeometryType = ThreeDCircularProjectionGeometry;
  using GeometryConstPointer = GeometryType::ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(DisplacedDetectorImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetConstObjectMacro(Geometry, GeometryType);
  itkSetConstObjectMacro(Geometry, GeometryType);

  /** Get / Set whether the projections should be padded
   * (yes for FDK, no for iterative) */
  itkGetMacro(PadOnTruncatedSide, bool);
  itkSetMacro(PadOnTruncatedSide, bool);

  /**
   * Get / Set the minimum and maximum offsets of the detector along the
   * weighting direction desribed in ToUntiltedCoordinate.
   */
  void
  SetOffsets(double minOffset, double maxOffset);
  itkGetMacro(MinimumOffset, double);
  itkGetMacro(MaximumOffset, double);

  /** Get / Set the Disable parameter
   */
  itkGetMacro(Disable, bool);
  itkSetMacro(Disable, bool);

protected:
  DisplacedDetectorImageFilter() = default;

  ~DisplacedDetectorImageFilter() override = default;

  /** Retrieve computed inferior and superior corners */
  itkGetMacro(InferiorCorner, double);
  itkGetMacro(SuperiorCorner, double);

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() const override;

  void
  GenerateInputRequestedRegion() override;

  void
  GenerateOutputInformation() override;

  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

  // Iterative filters do not need padding
  bool m_PadOnTruncatedSide{ true };

private:
  /** RTK geometry object */
  GeometryConstPointer m_Geometry;

  /**
   * Minimum and maximum offsets of the detector along the weighting direction, i.e. x.
   * If a priori known, these values can be given as input. Otherwise, they are computed from the
   * complete geometry.
   */
  double m_MinimumOffset{ 0. };
  double m_MaximumOffset{ 0. };

  /**
   * Flag used to know if the user has entered the min/max values of the detector offset.
   */
  bool m_OffsetsSet{ false };

  /** Superior and inferior position of the detector along the weighting
   *  direction, i.e., the virtual detector described in ToUntiltedCoordinate.
   */
  double m_InferiorCorner{ 0. };
  double m_SuperiorCorner{ 0. };

  /** When using a geometry that the displaced detector cannot manage,
   * it has to be disabled
   */
  bool m_Disable{ false };

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkDisplacedDetectorImageFilter.hxx"
#endif

#endif
