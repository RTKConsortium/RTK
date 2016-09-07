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

#ifndef rtkHilbertImageFilter_hxx
#define rtkHilbertImageFilter_hxx

#include <itkConfigure.h>
#include <itkForwardFFTImageFilter.h>
#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 7)
# include <itkComplexToComplexFFTImageFilter.h>
#endif
#include <itkImageRegionIteratorWithIndex.h>

namespace rtk
{

template<class TInputImage, class TOutputImage>
void
HilbertImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  // Take the FFT of the input
  typedef typename itk::ForwardFFTImageFilter< TInputImage, TOutputImage > FFTFilterType;
  typename FFTFilterType::Pointer fftFilt = FFTFilterType::New();
  fftFilt->SetInput( this->GetInput() );
  fftFilt->Update();

  TOutputImage *fft = fftFilt->GetOutput();

  // Weights according to
  // [Marple, IEEE Trans Sig Proc, 1999]
  typedef typename itk::ImageRegionIteratorWithIndex<TOutputImage> IteratorType;
  IteratorType it(fft, fft->GetLargestPossibleRegion());
  it.Set(it.Get());
  ++it;
  int n = fft->GetLargestPossibleRegion().GetSize()[0];
  for(int i=1; i<n/2-1; i++, ++it)
    it.Set( 2. * it.Get() );
  if(n%2 == 1) // Odd
    it.Set( 2. * it.Get() );
  else
    it.Set( 1. * it.Get() );
  typename TOutputImage::PixelType val = 0.;
  while( !it.IsAtEnd() )
    {
    val = it.Get();
    it.Set(0.);
    ++it;
    }

  // Inverse FFT (although I had to set it to FORWARD to obtain the same as in Matlab,
  // I really don't know why)
#if !defined(USE_FFTWD)
  if(typeid(typename TOutputImage::PixelType).name() == typeid(double).name() )
    {
    itkExceptionMacro(<< "FFTW with double has not been activated in ITK, cannot run.");
    }
#endif
#if !defined(USE_FFTWF)
  if(typeid(typename TOutputImage::PixelType).name() == typeid(float).name() )
    {
    itkExceptionMacro(<< "FFTW with float has not been activated in ITK, cannot run.");
    }
#endif

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 7)
  typedef typename itk::ComplexToComplexFFTImageFilter<TOutputImage> InverseFFTFilterType;
  typename InverseFFTFilterType::Pointer invFilt = InverseFFTFilterType::New();
  invFilt->SetTransformDirection(InverseFFTFilterType::FORWARD);
  invFilt->SetInput(fft);
  invFilt->Update();

  this->GraftOutput( invFilt->GetOutput() );
#else
  itkExceptionMacro(<< "Only available for ITK versions greater than 4.7.");
#endif
}

} // end of namespace rtk
#endif
