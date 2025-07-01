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

#ifndef rtkFFTHilbertImageFilter_hxx
#define rtkFFTHilbertImageFilter_hxx

#include "rtkFFTHilbertImageFilter.h"
#include "itkForwardFFTImageFilter.h"

namespace rtk
{

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTHilbertImageFilter<TInputImage, TOutputImage, TFFTPrecision>::GenerateOutputInformation()
{
  FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::GenerateOutputInformation();

  auto   origin = this->GetOutput()->GetOrigin();
  double spacing = this->GetOutput()->GetSpacing()[0];
  origin[0] += m_PixelShift * spacing;
  this->GetOutput()->SetOrigin(origin);
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTHilbertImageFilter<TInputImage, TOutputImage, TFFTPrecision>::UpdateFFTProjectionsConvolutionKernel(const SizeType s)
{
  if (this->m_KernelFFT.GetPointer() != nullptr && s == this->m_PreviousKernelUpdateSize)
  {
    return;
  }
  m_PreviousKernelUpdateSize = s;

  const int width = s[0];

  // Allocate kernel
  SizeType size;
  size.Fill(1);
  size[0] = width;
  FFTInputImagePointer kernel = FFTInputImageType::New();
  kernel->SetRegions(size);
  kernel->Allocate();
  kernel->FillBuffer(0.);

  itk::ImageRegionIterator<FFTInputImageType> it(kernel, kernel->GetLargestPossibleRegion());

  // Compute band-limited kernel in space domain
  double    spacing = this->GetInput()->GetSpacing()[0];
  IndexType ix;

  while (!it.IsAtEnd())
  {
    ix = it.GetIndex();

    double x = (ix[0] + m_PixelShift) * spacing;
    if (x > (size[0] / 2) * spacing)
      x -= size[0] * spacing;

    if (x == 0.)
    {
      it.Set(0.);
    }
    else
    {
      double v = spacing * (1. - std::cos(itk::Math::pi * x / spacing)) / (itk::Math::pi * x);
      it.Set(v);
    }

    ++it;
  }

  // FFT kernel
  using FFTType = itk::RealToHalfHermitianForwardFFTImageFilter<FFTInputImageType, FFTOutputImageType>;
  auto fftK = FFTType::New();
  fftK->SetInput(kernel);
  fftK->SetNumberOfWorkUnits(this->GetNumberOfWorkUnits());
  fftK->Update();

  this->m_KernelFFT = fftK->GetOutput();
  this->m_KernelFFT->DisconnectPipeline();
}

} // end namespace rtk
#endif
