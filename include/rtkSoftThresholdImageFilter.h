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

#ifndef rtkSoftThresholdImageFilter_h
#define rtkSoftThresholdImageFilter_h

#include "itkUnaryFunctorImageFilter.h"
#include "itkConceptChecking.h"
#include "itkSimpleDataObjectDecorator.h"

namespace rtk
{

/** \class SoftThresholdImageFilter
 *
 * \brief Soft thresholds an image
 *
 * This filter produces an output image whose pixels
 * are max(x-t,0).sign(x) where x is the corresponding
 * input pixel value and t the threshold
 *
 * \ingroup RTK IntensityImageFilters  Multithreaded
 */

namespace Functor
{

template <class TInput, class TOutput>
class ITK_TEMPLATE_EXPORT SoftThreshold
{
public:
  SoftThreshold() { m_Threshold = itk::NumericTraits<TInput>::Zero; }
  ~SoftThreshold() = default;

  void
  SetThreshold(const TInput & thresh)
  {
    m_Threshold = thresh;
  }

  bool
  operator!=(const SoftThreshold & other) const
  {
    if (m_Threshold != other.m_Threshold)
    {
      return true;
    }
    return false;
  }
  bool
  operator==(const SoftThreshold & other) const
  {
    return !(*this != other);
  }

  inline TOutput
  operator()(const TInput & A) const
  {
    return (itk::Math::sgn(A) * std::max((TInput)itk::Math::abs(A) - m_Threshold, (TInput)0.0));
  }

private:
  TInput m_Threshold;
};
} // namespace Functor

template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT SoftThresholdImageFilter
  : public itk::UnaryFunctorImageFilter<
      TInputImage,
      TOutputImage,
      Functor::SoftThreshold<typename TInputImage::PixelType, typename TOutputImage::PixelType>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(SoftThresholdImageFilter);

  /** Standard class type alias. */
  using Self = SoftThresholdImageFilter;
  typedef itk::UnaryFunctorImageFilter<
    TInputImage,
    TOutputImage,
    Functor::SoftThreshold<typename TInputImage::PixelType, typename TOutputImage::PixelType>>
    Superclass;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(SoftThresholdImageFilter);

  /** Pixel types. */
  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;

  /** Type of DataObjects to use for scalar inputs */
  using InputPixelObjectType = itk::SimpleDataObjectDecorator<InputPixelType>;

  /** Set the threshold */
  virtual void
  SetThreshold(const InputPixelType threshold);

  /** Begin concept checking */
  itkConceptMacro(OutputEqualityComparableCheck, (itk::Concept::EqualityComparable<OutputPixelType>));
  itkConceptMacro(InputPixelTypeComparable, (itk::Concept::Comparable<InputPixelType>));
  itkConceptMacro(InputOStreamWritableCheck, (itk::Concept::OStreamWritable<InputPixelType>));
  itkConceptMacro(OutputOStreamWritableCheck, (itk::Concept::OStreamWritable<OutputPixelType>));
  /** End concept checking */

protected:
  SoftThresholdImageFilter();
  ~SoftThresholdImageFilter() override = default;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkSoftThresholdImageFilter.hxx"
#endif

#endif
