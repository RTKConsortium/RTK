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

#ifndef rtkSumOfSquaresImageFilter_h
#define rtkSumOfSquaresImageFilter_h

#include <itkInPlaceImageFilter.h>

/** \class SumOfSquaresImageFilter
 * \brief Computes the sum of squared differences between two images
 *
 * Works on vector images too.
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
namespace rtk
{

template <class TOutputImage>
class ITK_TEMPLATE_EXPORT SumOfSquaresImageFilter : public itk::InPlaceImageFilter<TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(SumOfSquaresImageFilter);

  /** Standard class type alias. */
  using Self = SumOfSquaresImageFilter;
  using Superclass = itk::InPlaceImageFilter<TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using OutputPixelType = typename TOutputImage::PixelType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using OutputInternalPixelType = typename TOutputImage::InternalPixelType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
#ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(SumOfSquaresImageFilter);
#else
  itkTypeMacro(SumOfSquaresImageFilter, itk::InPlaceImageFilter);
#endif

  /** Macro to get the SSD */
  itkGetMacro(SumOfSquares, OutputInternalPixelType);

protected:
  SumOfSquaresImageFilter();
  ~SumOfSquaresImageFilter() override = default;

  void
  BeforeThreadedGenerateData();
  void
  ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, itk::ThreadIdType threadId) override;
  void
  AfterThreadedGenerateData();

  OutputInternalPixelType              m_SumOfSquares;
  std::vector<OutputInternalPixelType> m_VectorOfPartialSSs;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkSumOfSquaresImageFilter.hxx"
#endif

#endif
