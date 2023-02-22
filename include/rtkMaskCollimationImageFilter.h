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

#ifndef rtkMaskCollimationImageFilter_h
#define rtkMaskCollimationImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkConfiguration.h"
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

/** \class MaskCollimationImageFilter
 * \brief Mask out everything behind the jaws (typically in Ora file format,
 * i.e., the medPhoton scanner)
 *
 * \test rtkoratest
 *
 * \author Simon Rit
 *
 * \ingroup RTK InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT MaskCollimationImageFilter : public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MaskCollimationImageFilter);

  /** Standard class type alias. */
  using Self = MaskCollimationImageFilter;
  using Superclass = itk::InPlaceImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using OutputImageRegionType = typename TOutputImage::RegionType;
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  using GeometryPointer = typename GeometryType::Pointer;
  using FileNamesContainer = std::vector<std::string>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MaskCollimationImageFilter, itk::ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetModifiableObjectMacro(Geometry, GeometryType);
  itkSetObjectMacro(Geometry, GeometryType);

  /** Set the index of the rotation axis. The rotation axis is assumed to be one of the vector of the basis of the
   * coordinate system. The default is the y vector (index 1). */
  itkSetMacro(RotationAxisIndex, unsigned int);

protected:
  MaskCollimationImageFilter();
  ~MaskCollimationImageFilter() override = default;

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() ITKv5_CONST override;

  void
  BeforeThreadedGenerateData() override;

  /** Apply changes to the input image requested region. */
  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void
  VerifyInputInformation() const override
  {}

  // Set default parameters
  unsigned int m_RotationAxisIndex = 1; // y-axis

private:
  /** RTK geometry object */
  GeometryPointer m_Geometry;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkMaskCollimationImageFilter.hxx"
#endif

#endif
