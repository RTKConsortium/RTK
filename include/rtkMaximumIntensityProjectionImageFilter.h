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

#ifndef rtkMaximumIntensityProjectionImageFilter_h
#define rtkMaximumIntensityProjectionImageFilter_h

#include "rtkJosephForwardProjectionImageFilter.h"

namespace rtk
{

/** \class MaximumIntensityProjectionImageFilter
 * \brief MIP filter.
 *
 * Performs a MIP forward projection, i.e. calculation of a maximum intensity
 * step along the x-ray line.
 *
 * \test rtkmaximumintensityprojectiontest.cxx
 *
 * \author Mikhail Polkovnikov
 *
 * \ingroup RTK Projector
 */

template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT MaximumIntensityProjectionImageFilter
  : public JosephForwardProjectionImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MaximumIntensityProjectionImageFilter);

  /** Standard class type alias. */
  using Self = MaximumIntensityProjectionImageFilter;
  using Pointer = itk::SmartPointer<Self>;
  using Superclass = JosephForwardProjectionImageFilter<TInputImage, TOutputImage>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;
  using PointType = typename TInputImage::PointType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using CoordinateType = double;
  using VectorType = itk::Vector<CoordinateType, TInputImage::ImageDimension>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(MaximumIntensityProjectionImageFilter);

protected:
  MaximumIntensityProjectionImageFilter();
  ~MaximumIntensityProjectionImageFilter() override = default;

  void
  ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId) override;

#if !defined(ITK_WRAPPING_PARSER)
  static constexpr auto MaximumSumAlongRay = [](const ThreadIdType,
                                                Superclass &,
                                                OutputPixelType &    mipValue,
                                                const InputPixelType volumeValue,
                                                const VectorType &) {
    OutputPixelType tmp = static_cast<OutputPixelType>(volumeValue);
    if (tmp > mipValue)
    {
      mipValue = tmp;
    }
  };
  using MaximumSumAlongRayType = std::remove_const_t<decltype(MaximumSumAlongRay)>;
  static constexpr auto MaximumProjectedValueAccumulation = [](const ThreadIdType,
                                                               Superclass &,
                                                               const InputPixelType &  input,
                                                               OutputPixelType &       output,
                                                               const OutputPixelType & rayCastValue,
                                                               const VectorType &      stepInMM,
                                                               const PointType &,
                                                               const VectorType &,
                                                               const PointType &,
                                                               const PointType &) {
    OutputPixelType tmp = static_cast<OutputPixelType>(input);
    if (tmp < rayCastValue)
    {
      tmp = rayCastValue;
    }
    output = tmp * stepInMM.GetNorm();
  };
  using MaximumProjectedValueAccumulationType = std::remove_const_t<decltype(MaximumProjectedValueAccumulation)>;
#endif
};
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkMaximumIntensityProjectionImageFilter.hxx"
#endif

#endif
