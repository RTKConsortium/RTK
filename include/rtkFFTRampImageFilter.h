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

#ifndef rtkFFTRampImageFilter_h
#define rtkFFTRampImageFilter_h

#include <itkConceptChecking.h>
#include "rtkConfiguration.h"
#include "rtkFFTProjectionsConvolutionImageFilter.h"
#include "rtkMacro.h"

// The Set macro is redefined to clear the current FFT kernel when a parameter
// is modified.
// clang-format off
#ifndef ITK_GCC_PRAGMA_PUSH
#define ITK_GCC_PRAGMA_PUSH CLANG_PRAGMA_PUSH
#endif
#ifndef ITK_GCC_SUPPRESS_Wfloat_equal
#define ITK_GCC_SUPPRESS_Wfloat_equal CLANG_SUPPRESS_Wfloat_equal
#endif
#ifndef ITK_GCC_PRAGMA_POP
#define ITK_GCC_PRAGMA_POP CLANG_PRAGMA_POP
#endif
#undef itkSetMacro
#define itkSetMacro(name, type)                     \
  virtual void Set##name(type _arg)                 \
  {                                                 \
    itkDebugMacro("setting " #name " to " << _arg); \
    ITK_GCC_PRAGMA_PUSH                                 \
    ITK_GCC_SUPPRESS_Wfloat_equal                       \
    if (this->m_##name != _arg)                     \
    {                                               \
      this->m_##name = std::move(_arg);             \
      this->Modified();                             \
      this->m_KernelFFT = nullptr;                  \
    }                                               \
    ITK_GCC_PRAGMA_POP                                  \
  }                                                 \
  ITK_MACROEND_NOOP_STATEMENT
// clang-format on

namespace rtk
{

/** \class FFTRampImageFilter
 * \brief Implements the ramp image filter of the filtered backprojection algorithm.
 *
 * The filter code is based on FFTProjectionsConvolutionImageFilter by Gaetan Lehmann
 * (see https://hdl.handle.net/10380/3154)
 *
 * \test rtkrampfiltertest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK ImageToImageFilter
 */

template <class TInputImage, class TOutputImage = TInputImage, class TFFTPrecision = double>
class ITK_TEMPLATE_EXPORT FFTRampImageFilter
  : public rtk::FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(FFTRampImageFilter);

  /** Standard class type alias. */
  using Self = FFTRampImageFilter;
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

  /** Runtime information support. */
#ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(FFTRampImageFilter);
#else
  itkTypeMacro(FFTRampImageFilter, FFTProjectionsConvolutionImageFilter);
#endif

  /** Set/Get the Hann window frequency. 0 (default) disables it */
  itkGetConstMacro(HannCutFrequency, double);
  itkSetMacro(HannCutFrequency, double);

  /** Set/Get the Cosine Cut window frequency. 0 (default) disables it */
  itkGetConstMacro(CosineCutFrequency, double);
  itkSetMacro(CosineCutFrequency, double);

  /** Set/Get the Hamming window frequency. 0 (default) disables it */
  itkGetConstMacro(HammingFrequency, double);
  itkSetMacro(HammingFrequency, double);

  /** Set/Get the Hann window frequency in Y direction. 0 (default) disables it */
  itkGetConstMacro(HannCutFrequencyY, double);
  itkSetMacro(HannCutFrequencyY, double);

  /** Set/Get the Ram-Lak window frequency (0...1). 0 (default) disable it.
   * Equation and further explanation about Ram-Lak filter could be found in:
   * 1. Fundamentals of 2D and 3D reconstruction (by Dr. Gunter Lauritsch).
   *    https://campar.in.tum.de/twiki/pub/Chair/TeachingWs04IOIV/08CTReconstruction.pdf
   * 2. Reconstruction.
   *    http://oftankonyv.reak.bme.hu/tiki-index.php?page=Reconstruction
   */
  itkGetConstMacro(RamLakCutFrequency, double);
  itkSetMacro(RamLakCutFrequency, double);

  /** Set/Get the Shepp-Logan window frequency (0...1). 0 (default) disable it.
   * Equation and further explanation about Shepp-Logan filter could be found in:
   * 1. Fundamentals of 2D and 3D reconstruction (by Dr. Gunter Lauritsch).
   *    https://campar.in.tum.de/twiki/pub/Chair/TeachingWs04IOIV/08CTReconstruction.pdf
   * 2. Reconstruction.
   *    http://oftankonyv.reak.bme.hu/tiki-index.php?page=Reconstruction
   */
  itkGetConstMacro(SheppLoganCutFrequency, double);
  itkSetMacro(SheppLoganCutFrequency, double);

protected:
  FFTRampImageFilter();
  ~FFTRampImageFilter() override = default;

  void
  GenerateInputRequestedRegion() override;

  /** Creates and return a pointer to one line of the ramp kernel in Fourier space.
   *  Used in generate data functions.  */
  void
  UpdateFFTProjectionsConvolutionKernel(const SizeType s) override;

  virtual void
  SquareKernel()
  {}

private:
  /**
   * Cut frequency of Hann, Cosine and Hamming windows. The first one which is
   * non-zero is used.
   */
  double m_HannCutFrequency{ 0. };
  double m_CosineCutFrequency{ 0. };
  double m_HammingFrequency{ 0. };
  double m_HannCutFrequencyY{ 0. };

  /** Cut frequency of Ram-Lak and Shepp-Logan
   */
  double m_RamLakCutFrequency{ 0. };
  double m_SheppLoganCutFrequency{ 0. };

  SizeType m_PreviousKernelUpdateSize;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkFFTRampImageFilter.hxx"
#endif

// Rollback to the original definition of the Set macro
// clang-format off
#undef itkSetMacro
#define itkSetMacro(name, type)                     \
  virtual void Set##name(type _arg)                 \
  {                                                 \
    itkDebugMacro("setting " #name " to " << _arg); \
    ITK_GCC_PRAGMA_PUSH                                 \
    ITK_GCC_SUPPRESS_Wfloat_equal                       \
    if (this->m_##name != _arg)                     \
    {                                               \
      this->m_##name = std::move(_arg);             \
      this->Modified();                             \
    }                                               \
    ITK_GCC_PRAGMA_POP                                  \
  }                                                 \
  ITK_MACROEND_NOOP_STATEMENT
// clang-format on
#endif
