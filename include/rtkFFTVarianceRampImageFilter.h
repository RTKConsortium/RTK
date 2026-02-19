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

#ifndef rtkFFTVarianceRampImageFilter_h
#define rtkFFTVarianceRampImageFilter_h

#include "rtkFFTRampImageFilter.h"
#include "rtkFFTProjectionsConvolutionImageFilter.h"
#include "itkHalfHermitianToRealInverseFFTImageFilter.h"

namespace rtk
{

/** \class FFTVarianceRampImageFilter
 * \brief Implements the variance image filter of the filtered backprojection algorithm.
 *
 * \test rtkvariancereconstructiontest
 *
 * \author Simon Rit
 * \ingroup RTK ImageToImageFilter
 */

template <class TInputImage, class TOutputImage = TInputImage, class TFFTPrecision = double>
class ITK_EXPORT FFTVarianceRampImageFilter : public rtk::FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
{
public:
  using Baseclass = rtk::FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>;

  /** Standard class typedefs. */
  using Self = FFTVarianceRampImageFilter;
  using Superclass = rtk::FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient typedefs. */
  using InputImageType = typename Baseclass::InputImageType;
  using OutputImageType = typename Baseclass::OutputImageType;
  using FFTPrecisionType = typename Baseclass::FFTPrecisionType;
  using IndexType = typename Baseclass::IndexType;
  using SizeType = typename Baseclass::SizeType;

  using FFTInputImageType = typename Baseclass::FFTInputImageType;
  using FFTInputImagePointer = typename Baseclass::FFTInputImagePointer;
  using FFTOutputImageType = typename Baseclass::FFTOutputImageType;
  using FFTOutputImagePointer = typename Baseclass::FFTOutputImagePointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(FFTVarianceRampImageFilter);

protected:
  FFTVarianceRampImageFilter();
  ~FFTVarianceRampImageFilter() = default;

  /** Creates and return a pointer to one line of the variance kernel in Fourier space.
   *  Used in generate data functions.  */
  void
  UpdateFFTProjectionsConvolutionKernel(const SizeType size) override;

  FFTVarianceRampImageFilter(const Self &); // purposely not implemented
  void
  operator=(const Self &); // purposely not implemented
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkFFTVarianceRampImageFilter.hxx"
#endif

#endif
