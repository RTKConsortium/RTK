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

#ifndef rtkFFTVarianceRampImageFilter_hxx
#define rtkFFTVarianceRampImageFilter_hxx

#include "rtkFFTVarianceRampImageFilter.h"

namespace rtk
{

template <class TInputImage, class TOutputImage, class TFFTPrecision>
FFTVarianceRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>::FFTVarianceRampImageFilter() = default;

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTVarianceRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>::UpdateFFTProjectionsConvolutionKernel(
  const SizeType s)
{
  if (this->GetHannCutFrequencyY() > 0.)
  {
    itkGenericExceptionMacro(<< "Variance Ramp image filter is not implemented for HannCutFrequencyY>0.");
  }

  if (this->m_KernelFFT.GetPointer() != nullptr && s == this->m_PreviousKernelUpdateSize)
  {
    return;
  }

  // Calculate the conventional ramp image filter Kernel
  Superclass::UpdateFFTProjectionsConvolutionKernel(s);

  // iFFT kernel
  auto ifftK = itk::HalfHermitianToRealInverseFFTImageFilter<FFTOutputImageType, FFTInputImageType>::New();
  ifftK->SetInput(this->m_KernelFFT);
  ifftK->SetNumberOfWorkUnits(this->GetNumberOfWorkUnits());
  ifftK->Update();
  using FFTType = itk::RealToHalfHermitianForwardFFTImageFilter<FFTInputImageType, FFTOutputImageType>;
  typename FFTType::InputImageType::Pointer KernelIFFT = ifftK->GetOutput();

  // calculate ratio g_C/g^2
  itk::ImageRegionIteratorWithIndex<typename FFTType::InputImageType> itiK(KernelIFFT,
                                                                           KernelIFFT->GetLargestPossibleRegion());
  typename FFTType::InputImageType::PixelType                         sumgc = 0.;
  typename FFTType::InputImageType::PixelType                         sumg2 = 0.;
  typename FFTType::InputImageType::IndexType                         idxshifted;
  const unsigned int widthFFT = KernelIFFT->GetLargestPossibleRegion().GetSize()[0];
  for (; !itiK.IsAtEnd(); ++itiK)
  {
    idxshifted = itiK.GetIndex();
    if (idxshifted[0] == 0)
      idxshifted[0] = widthFFT - 1;
    else
      idxshifted[0] -= 1;

    sumgc += itiK.Get() * KernelIFFT->GetPixel(idxshifted);
    sumg2 += itiK.Get() * itiK.Get();
  }
  const typename FFTType::InputImageType::PixelType ratiogcg2 = sumgc / sumg2;

  // numerical integration to calculate f_interp
  const double aprecision = 0.00001;
  double       finterp = 0.;
  for (unsigned int i = 0; i < int(1. / aprecision); i++)
  {
    const double a = double(i) * aprecision;
    finterp += (1 - a) * (1 - a) + 2 * ratiogcg2 * (1 - a) * a + a * a;
  }
  finterp *= aprecision;

  // square kernel and multiply with finterp
  itiK.GoToBegin();
  for (; !itiK.IsAtEnd(); ++itiK)
  {
    itiK.Set(itiK.Get() * itiK.Get() * finterp);
  }

  // FFT kernel
  auto fftK2 = FFTType::New();
  fftK2->SetInput(KernelIFFT);
  fftK2->SetNumberOfWorkUnits(this->GetNumberOfWorkUnits());
  fftK2->Update();
  this->m_KernelFFT = fftK2->GetOutput();
  this->m_KernelFFT->DisconnectPipeline();
}

} // end namespace rtk
#endif
