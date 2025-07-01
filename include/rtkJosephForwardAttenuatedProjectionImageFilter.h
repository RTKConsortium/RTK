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

#ifndef rtkJosephForwardAttenuatedProjectionImageFilter_h
#define rtkJosephForwardAttenuatedProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkMacro.h"
#include <itkPixelTraits.h>
#include <cmath>
#include <vector>

namespace rtk
{

/** \class JosephForwardAttenuatedProjectionImageFilter
 * \brief Joseph forward projection.
 *
 * Performs a attenuated Joseph forward projection, i.e. accumulation along x-ray lines,
 * using [Joseph, IEEE TMI, 1982] and [Gullberg, Phys. Med. Biol., 1985]. The forward projector tests if the  detector
 * has been placed after the source and the volume. If the detector is in the volume
 * the ray tracing is performed only until that point.
 *
 * \test rtkforwardattenuatedprojectiontest.cxx
 *
 * \author Antoine Robert
 *
 * \ingroup RTK Projector
 */

template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT JosephForwardAttenuatedProjectionImageFilter
  : public JosephForwardProjectionImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(JosephForwardAttenuatedProjectionImageFilter);

  /** Standard class type alias. */
  using Self = JosephForwardAttenuatedProjectionImageFilter;
  using Superclass = JosephForwardProjectionImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using CoordinateType = double;
  using VectorType = itk::Vector<CoordinateType, TInputImage::ImageDimension>;
  using WeightCoordinateType = typename itk::PixelTraits<InputPixelType>::ValueType;

  /** ImageDimension constants */
  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(JosephForwardAttenuatedProjectionImageFilter);

protected:
  JosephForwardAttenuatedProjectionImageFilter();
  ~JosephForwardAttenuatedProjectionImageFilter() override = default;

  /** Apply changes to the input image requested region. */
  void
  GenerateInputRequestedRegion() override;

  void
  BeforeThreadedGenerateData() override;

  /** Only the last two inputs should be in the same space so we need
   * to overwrite the method. */
  void
  VerifyInputInformation() const override;

  std::ptrdiff_t                                   m_AttenuationMinusEmissionMapsPtrDiff;
  std::array<InputPixelType, itk::ITK_MAX_THREADS> m_AttenuationRay;
  std::array<InputPixelType, itk::ITK_MAX_THREADS> m_AttenuationPixel;
  std::array<InputPixelType, itk::ITK_MAX_THREADS> m_Ex1;
};
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkJosephForwardAttenuatedProjectionImageFilter.hxx"
#endif

#endif
