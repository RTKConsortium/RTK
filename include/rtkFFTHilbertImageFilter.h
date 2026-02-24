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

#ifndef rtkFFTHilbertImageFilter_h
#define rtkFFTHilbertImageFilter_h

#include "rtkConfiguration.h"
#include "rtkFFTProjectionsConvolutionImageFilter.h"
#include "rtkMacro.h"
#include <itkConceptChecking.h>

namespace rtk
{

/** \class FFTHilbertImageFilter
 * \brief Implements the Hilbert transform.
 *
 * \test rtkhilbertfiltertest.cxx
 *
 * \author Aurélien Coussat
 *
 * \ingroup RTK ImageToImageFilter
 */

template <class TInputImage, class TOutputImage = TInputImage, class TFFTPrecision = double>
class ITK_EXPORT FFTHilbertImageFilter
  : public rtk::FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(FFTHilbertImageFilter);

  /** Standard class type alias. */
  using Self = FFTHilbertImageFilter;
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

  /** Standard New method. */
  itkNewMacro(Self);

  itkGetMacro(PixelShift, double);
  // The Set macro is redefined to clear the current FFT kernel when a parameter
  // is modified.
  virtual void
  SetPixelShift(const double _arg)
  {
    itkDebugMacro("setting PixelShift to " << _arg);
    CLANG_PRAGMA_PUSH
    CLANG_SUPPRESS_Wfloat_equal
    if (this->m_PixelShift != _arg)
    {
      this->m_PixelShift = _arg;
      this->Modified();
      this->m_KernelFFT = nullptr;
    }
    CLANG_PRAGMA_POP
  }

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(FFTHilbertImageFilter);

protected:
  FFTHilbertImageFilter() = default;
  ~FFTHilbertImageFilter() override = default;

  void
  GenerateOutputInformation() override;

  /** Create and return a pointer to one line of the Hilbert kernel in Fourier space.
   *  Used in generate data functions.  */
  void
  UpdateFFTProjectionsConvolutionKernel(SizeType s) override;

private:
  SizeType m_PreviousKernelUpdateSize;
  double   m_PixelShift{};

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkFFTHilbertImageFilter.hxx"
#endif

#endif
