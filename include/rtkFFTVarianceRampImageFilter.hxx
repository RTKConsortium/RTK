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

#ifndef rtkFFTVarianceRampImageFilter_hxx
#define rtkFFTVarianceRampImageFilter_hxx
 
namespace rtk
{

template <class TInputImage, class TOutputImage, class TFFTPrecision>
FFTVarianceRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>::FFTVarianceRampImageFilter() = default;

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTVarianceRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::UpdateFFTProjectionsConvolutionKernel(const SizeType s)
{
  if(this->m_KernelFFT.GetPointer() != nullptr && s == this->GetPreviousKernelUpdateSize())
   {
   return;
   }
  this->SetPreviousKernelUpdateSize(s);

  const int width = s[0];
  const int height = s[1];

  // Allocate kernel
  SizeType size;
  size.Fill(1);
  size[0] = width;
  FFTInputImagePointer kernel = FFTInputImageType::New();
  kernel->SetRegions( size );
  kernel->Allocate();
  kernel->FillBuffer(0.);

  // Compute kernel in space domain (see Kak & Slaney, chapter 3 equation 61
  // page 72) although spacing is not squared according to equation 69 page 75
  double spacing = this->GetInput()->GetSpacing()[0];
  IndexType ix,jx;
  ix.Fill(0);
  jx.Fill(0);
  kernel->SetPixel(ix, 1./(4.*spacing) );
  for(ix[0]=1, jx[0]=size[0]-1; ix[0] < typename IndexType::IndexValueType(size[0]/2); ix[0] += 2, jx[0] -= 2)
    {
    double v = ix[0] * itk::Math::pi;
    v = -1. / (v * v * spacing);
    kernel->SetPixel(ix, v);
    kernel->SetPixel(jx, v);
    }

  // FFT kernel
  using FFTType = itk::RealToHalfHermitianForwardFFTImageFilter< FFTInputImageType, FFTOutputImageType >;
  typename FFTType::Pointer fftK = FFTType::New();
  fftK->SetInput( kernel );
  fftK->SetNumberOfWorkUnits( this->GetNumberOfWorkUnits() );
  fftK->Update();
  this->m_KernelFFT = fftK->GetOutput();

  // Windowing (if enabled)
  using IteratorType = itk::ImageRegionIteratorWithIndex<typename FFTType::OutputImageType>;
  IteratorType itK(this->m_KernelFFT, this->m_KernelFFT->GetLargestPossibleRegion() );

  unsigned int n = this->m_KernelFFT->GetLargestPossibleRegion().GetSize(0);

  itK.GoToBegin();
  if (this->GetHannCutFrequency() > 0.)
  {
    const unsigned int ncut = itk::Math::Round<double>(n * std::min(1.0, this->GetHannCutFrequency()));
    for (unsigned int i = 0; i < ncut; i++, ++itK)
      itK.Set(itK.Get() * TFFTPrecision(0.5 * (1 + std::cos(itk::Math::pi * i / ncut))));
  }
  else if (this->GetCosineCutFrequency() > 0.)
  {
    const unsigned int ncut = itk::Math::Round<double>(n * std::min(1.0, this->GetCosineCutFrequency()));
    for (unsigned int i = 0; i < ncut; i++, ++itK)
      itK.Set(itK.Get() * TFFTPrecision(std::cos(0.5 * itk::Math::pi * i / ncut)));
  }
  else if (this->GetHammingFrequency() > 0.)
  {
    const unsigned int ncut = itk::Math::Round<double>(n * std::min(1.0, this->GetHammingFrequency()));
    for (unsigned int i = 0; i < ncut; i++, ++itK)
      itK.Set(itK.Get() * TFFTPrecision(0.54 + 0.46 * (std::cos(itk::Math::pi * i / ncut))));
  }
  else if (this->GetRamLakCutFrequency() > 0.)
  {
    const unsigned int ncut = itk::Math::Round<double>(n * std::min(1.0, this->GetRamLakCutFrequency()));
    for (unsigned int i = 0; i < ncut; i++, ++itK)
    {
    }
  }
  else if (this->GetSheppLoganCutFrequency() > 0.)
  {
    const unsigned int ncut = itk::Math::Round<double>(n * std::min(1.0, this->GetSheppLoganCutFrequency()));
    // sinc(0) --> is 1
    ++itK;
    for (unsigned int i = 1; i < ncut; i++, ++itK)
    {
      double x = 0.5 * itk::Math::pi * i / ncut;
      itK.Set(itK.Get() * TFFTPrecision(std::sin(x) / x));
    }
  }
  else
  {
    itK.GoToReverseBegin();
    ++itK;
  }

  for(; !itK.IsAtEnd(); ++itK)
    {
    itK.Set( itK.Get() * TFFTPrecision(0.) );
    }

  // iFFT kernel
  using IFFTType = itk::HalfHermitianToRealInverseFFTImageFilter< FFTOutputImageType, FFTInputImageType >;
  typename IFFTType::Pointer ifftK = IFFTType::New();
  ifftK->SetInput( this->m_KernelFFT );
  ifftK->SetNumberOfThreads( this->GetNumberOfThreads() );
  ifftK->Update();
  typename FFTType::InputImageType::Pointer KernelIFFT = ifftK->GetOutput();

  // calculate ratio g_C/g^2
  using InverseIteratorType = itk::ImageRegionIteratorWithIndex<typename FFTType::InputImageType>;
  InverseIteratorType itiK(KernelIFFT, KernelIFFT->GetLargestPossibleRegion());
  typename FFTType::InputImageType::PixelType sumgc = 0.;
  typename FFTType::InputImageType::PixelType sumg2 = 0.;
  typename FFTType::InputImageType::IndexType idxshifted;
  unsigned int const widthFFT = KernelIFFT->GetLargestPossibleRegion().GetSize()[0];
  for(; !itiK.IsAtEnd(); ++itiK)
  {
    idxshifted = itiK.GetIndex();
    if(idxshifted[0] == 0)
      idxshifted[0] = widthFFT-1;
    else
      idxshifted[0] -= 1;

    sumgc += itiK.Get() * KernelIFFT->GetPixel(idxshifted);
    sumg2 += itiK.Get() * itiK.Get();
  }
  typename FFTType::InputImageType::PixelType const ratiogcg2 = sumgc/sumg2;

  // numerical integration to calculate f_interp
  double const aprecision = 0.00001;
  double finterp = 0.;
  for(unsigned int i=0; i<int(1./aprecision); i++)
    {
    double const a = double(i)*aprecision;
    finterp += (1-a)*(1-a) + 2*ratiogcg2*(1-a)*a + a*a;
    }
  finterp *= aprecision;

  // square kernel and multiply with finterp
  itiK.GoToBegin();
  for(; !itiK.IsAtEnd(); ++itiK)
  {
    itiK.Set(itiK.Get() * itiK.Get() * finterp);
  }

  // FFT kernel
  typename FFTType::Pointer fftK2 = FFTType::New();
  fftK2->SetInput( ifftK->GetOutput() );
  fftK2->SetNumberOfThreads( this->GetNumberOfThreads() );
  fftK2->Update();
  this->m_KernelFFT = fftK2->GetOutput();

  // Replicate and window if required
  if(this->GetHannCutFrequencyY()>0.)
    {
    std::cerr << "HannY not implemented for variance image filter." << std::endl;

    size.Fill(1);
    size[0] = this->m_KernelFFT->GetLargestPossibleRegion().GetSize(0);
    size[1] = height;

    const unsigned int ncut = itk::Math::Round<double>( (height/2+1) * std::min(1.0, this->GetHannCutFrequencyY() ) );

    this->m_KernelFFT = FFTOutputImageType::New();
    this->m_KernelFFT->SetRegions( size );
    this->m_KernelFFT->Allocate();
    this->m_KernelFFT->FillBuffer(0.);

    IteratorType itTwoDK(this->m_KernelFFT, this->m_KernelFFT->GetLargestPossibleRegion() );
    for(unsigned int j=0; j<ncut; j++)
      {
      itK.GoToBegin();
      const TFFTPrecision win( 0.5*( 1+std::cos(itk::Math::pi*j/ncut) ) );
      for(unsigned int i=0; i<size[0]; ++itK, ++itTwoDK, i++)
        {
        itTwoDK.Set( win * itK.Get() );
        }
      }
    itTwoDK.GoToReverseBegin();
    for(unsigned int j=1; j<ncut; j++)
      {
      itK.GoToReverseBegin();
      const TFFTPrecision win( 0.5*( 1+std::cos(itk::Math::pi*j/ncut) ) );
      for(unsigned int i=0; i<size[0]; --itK, --itTwoDK, i++)
        {
        itTwoDK.Set( win * itK.Get() );
        }
      }
    }
  this->m_KernelFFT->DisconnectPipeline();
}

} // end namespace rtk
#endif
