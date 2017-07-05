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

#ifndef rtkFFTRampImageFilter_h
#define rtkFFTRampImageFilter_h

#include <itkConceptChecking.h>
#include "rtkConfiguration.h"
#include "rtkFFTConvolutionImageFilter.h"
#include "rtkMacro.h"

// The Set macro is redefined to clear the current FFT kernel when a parameter
// is modified.
#undef itkSetMacro
#define itkSetMacro(name, type)                     \
  virtual void Set##name (const type _arg)          \
    {                                               \
    itkDebugMacro("setting " #name " to " << _arg); \
CLANG_PRAGMA_PUSH                                   \
CLANG_SUPPRESS_Wfloat_equal                         \
    if ( this->m_##name != _arg )                   \
      {                                             \
      this->m_##name = _arg;                        \
      this->Modified();                             \
      this->m_KernelFFT = ITK_NULLPTR;              \
      }                                             \
CLANG_PRAGMA_POP                                    \
    }

namespace rtk
{

/** \class FFTRampImageFilter
 * \brief Implements the ramp image filter of the filtered backprojection algorithm.
 *
 * The filter code is based on FFTConvolutionImageFilter by Gaetan Lehmann
 * (see http://hdl.handle.net/10380/3154)
 *
 * \test rtkrampfiltertest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */

template<class TInputImage, class TOutputImage=TInputImage, class TFFTPrecision=double>
class ITK_EXPORT FFTRampImageFilter :
  public rtk::FFTConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
{
public:
  /** Standard class typedefs. */
  typedef FFTRampImageFilter                                 Self;
  typedef rtk::FFTConvolutionImageFilter< TInputImage,
                                          TOutputImage,
                                          TFFTPrecision>     Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                                       InputImageType;
  typedef TOutputImage                                      OutputImageType;
  typedef TFFTPrecision                                     FFTPrecisionType;
  typedef typename InputImageType::IndexType                IndexType;
  typedef typename InputImageType::SizeType                 SizeType;

  typedef typename Superclass::FFTInputImageType            FFTInputImageType;
  typedef typename FFTInputImageType::Pointer               FFTInputImagePointer;
  typedef typename Superclass::FFTOutputImageType           FFTOutputImageType;
  typedef typename FFTOutputImageType::Pointer              FFTOutputImagePointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(FFTRampImageFilter, FFTConvolutionImageFilter);

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
   * 1. Fundamentals of 2D and 3D reconstruction (by Dr. Günter Lauritsch). 
   *    http://campar.in.tum.de/twiki/pub/Chair/TeachingWs04IOIV/08CTReconstruction.pdf
   * 2. Reconstruction. 
   *    http://oftankonyv.reak.bme.hu/tiki-index.php?page=Reconstruction
   */
  itkGetConstMacro(RamLakCutFrequency, double);
  itkSetMacro(RamLakCutFrequency, double);
  
  /** Set/Get the Shepp-Logan window frequency (0...1). 0 (default) disable it.
   * Equation and further explanation about Shepp-Logan filter could be found in:
   * 1. Fundamentals of 2D and 3D reconstruction (by Dr. Günter Lauritsch). 
   *    http://campar.in.tum.de/twiki/pub/Chair/TeachingWs04IOIV/08CTReconstruction.pdf
   * 2. Reconstruction. 
   *    http://oftankonyv.reak.bme.hu/tiki-index.php?page=Reconstruction
   */
  itkGetConstMacro(SheppLoganCutFrequency, double);
  itkSetMacro(SheppLoganCutFrequency, double);
  
protected:
  FFTRampImageFilter();
  ~FFTRampImageFilter() {}

  virtual void GenerateInputRequestedRegion() ITK_OVERRIDE;

  /** Creates and return a pointer to one line of the ramp kernel in Fourier space.
   *  Used in generate data functions.  */
  void UpdateFFTConvolutionKernel(const SizeType size) ITK_OVERRIDE;

private:
  FFTRampImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);     //purposely not implemented

  /**
   * Cut frequency of Hann, Cosine and Hamming windows. The first one which is
   * non-zero is used.
   */
  double m_HannCutFrequency;
  double m_CosineCutFrequency;
  double m_HammingFrequency;
  double m_HannCutFrequencyY;
  
  /** Cut frequency of Ram-Lak and Shepp-Logan 
    */
  double m_RamLakCutFrequency;
  double m_SheppLoganCutFrequency;

  SizeType m_PreviousKernelUpdateSize;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFFTRampImageFilter.hxx"
#endif

// Rollback to the original definition of the Set macro
#undef itkSetMacro
#define itkSetMacro(name, type)                     \
  virtual void Set##name (const type _arg)          \
    {                                               \
    itkDebugMacro("setting " #name " to " << _arg); \
CLANG_PRAGMA_PUSH                                   \
CLANG_SUPPRESS_Wfloat_equal                         \
    if ( this->m_##name != _arg )                   \
      {                                             \
      this->m_##name = _arg;                        \
      this->Modified();                             \
      }                                             \
CLANG_PRAGMA_POP                                    \
    }

#endif
