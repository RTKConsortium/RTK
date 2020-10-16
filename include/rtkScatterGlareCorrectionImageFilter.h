/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkScatterGlareCorrectionImageFilter_h
#define rtkScatterGlareCorrectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkFFTProjectionsConvolutionImageFilter.h"

namespace rtk
{

/** \class ScatterGlareCorrectionImageFilter
 * \brief Implements the scatter glare correction as described in [Poludniowski, PMB 2011].
 *
 * The filter code is based on FFTConvolutionImageFilter by Gaetan Lehmann
 * (see http://hdl.handle.net/10380/3154)
 *
 * \test rtkscatterglaretest.cxx
 *
 * \author Sebastien Brousmiche
 *
 * \ingroup RTK ImageToImageFilter
 */

template <class TInputImage, class TOutputImage = TInputImage, class TFFTPrecision = double>
class ITK_EXPORT ScatterGlareCorrectionImageFilter
  : public rtk::FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(ScatterGlareCorrectionImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(ScatterGlareCorrectionImageFilter);
#endif

  /** Standard class type alias. */
  using Self = ScatterGlareCorrectionImageFilter;
  using Superclass = rtk::FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using FFTPrecisionType = TFFTPrecision;
  using IndexType = typename InputImageType::IndexType;
  using SizeType = typename InputImageType::SizeType;

  using FFTInputImageType = typename Superclass::FFTInputImageType;
  using FFTInputImagePointer = typename FFTInputImageType::Pointer;
  using FFTOutputImageType = typename Superclass::FFTOutputImageType;
  using FFTOutputImagePointer = typename FFTOutputImageType::Pointer;

  using CoefficientVectorType = typename std::vector<float>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ScatterGlareCorrectionImageFilter, FFTProjectionsConvolutionImageFilter);

  itkGetConstMacro(Coefficients, CoefficientVectorType);
  virtual void
  SetCoefficients(const CoefficientVectorType coefficients)
  {
    if (this->m_Coefficients != coefficients)
    {
      this->m_Coefficients = coefficients;
      this->Modified();
    }
  }

protected:
  ScatterGlareCorrectionImageFilter();
  ~ScatterGlareCorrectionImageFilter() override = default;

  /** Create the deconvolution kernel
   */
  void
  UpdateFFTProjectionsConvolutionKernel(const SizeType size) override;

private:
  CoefficientVectorType m_Coefficients;
  CoefficientVectorType m_PreviousCoefficients;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkScatterGlareCorrectionImageFilter.hxx"
#endif

#endif
