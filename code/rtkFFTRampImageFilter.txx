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

#ifndef __rtkFFTRampImageFilter_txx
#define __rtkFFTRampImageFilter_txx

// Use local RTK FFTW files taken from Gaëtan Lehmann's code for
// thread safety: http://hdl.handle.net/10380/3154
#if defined(USE_FFTWD) || defined(USE_FFTWF)
#  include "itkFFTWRealToComplexConjugateImageFilter.h"
#  include "itkFFTWComplexConjugateToRealImageFilter.h"
#endif
#include <itkFFTRealToComplexConjugateImageFilter.h>
#include <itkFFTComplexConjugateToRealImageFilter.h>

#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

namespace rtk
{

template <class TInputImage, class TOutputImage, class TFFTPrecision>
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::FFTRampImageFilter() :
  m_TruncationCorrection(0.), m_GreatestPrimeFactor(2), m_HannCutFrequency(0.)
  ,m_CosineCutFrequency(0.),m_HammingFrequency(0.)
{
#if defined(USE_FFTWD)
  if(typeid(TFFTPrecision).name() == std::string("double") )
    m_GreatestPrimeFactor = 13;
#endif
#if defined(USE_FFTWF)
  if(typeid(TFFTPrecision).name() == std::string("float") )
    m_GreatestPrimeFactor = 13;
#endif
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  InputImageType * input = const_cast<InputImageType *>(this->GetInput() );
  if ( !input )
    return;

  // Compute input region (==requested region fully enlarged for dim 0)
  RegionType inputRegion;
  this->CallCopyOutputRegionToInputRegion(inputRegion, this->GetOutput()->GetRequestedRegion() );
  input->SetRequestedRegion( inputRegion );
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
int
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::GetTruncationCorrectionExtent()
{
  return itk::Math::Floor<TFFTPrecision>(m_TruncationCorrection * this->GetInput()->GetRequestedRegion().GetSize(0) );
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::BeforeThreadedGenerateData()
{
  UpdateTruncationMirrorWeights();

  // Force init of fftw library mutex (static class member) before
  // multithreading
#if defined(USE_FFTWF)
  itk::fftw::Proxy<float>::Lock();
  itk::fftw::Proxy<float>::Unlock();
#endif
#if defined(USE_FFTWD)
  itk::fftw::Proxy<double>::Lock();
  itk::fftw::Proxy<double>::Unlock();
#endif
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::ThreadedGenerateData( const RegionType& outputRegionForThread, ThreadIdType threadId )
{
  // Pad image region
  FFTInputImagePointer paddedImage = PadInputImageRegion(outputRegionForThread);

  // FFT padded image
#if ITK_VERSION_MAJOR <= 3
  typedef itk::FFTRealToComplexConjugateImageFilter< FFTPrecisionType, ImageDimension > FFTType;
#else
  typedef itk::FFTRealToComplexConjugateImageFilter< FFTInputImageType > FFTType;
#endif
  typename FFTType::Pointer fftI = FFTType::New();
  fftI->SetInput( paddedImage );
  fftI->SetNumberOfThreads( 1 );
  fftI->Update();

  // Get FFT ramp kernel
  FFTOutputImagePointer fftK = this->GetFFTRampKernel(paddedImage->GetLargestPossibleRegion().GetSize(0) );

  //Multiply line-by-line
  itk::ImageRegionIterator<typename FFTType::OutputImageType> itI(fftI->GetOutput(),
                                                              fftI->GetOutput()->GetLargestPossibleRegion() );
  itk::ImageRegionConstIterator<FFTOutputImageType> itK(fftK, fftK->GetLargestPossibleRegion() );
  itI.GoToBegin();
  while(!itI.IsAtEnd() ) {
    itK.GoToBegin();
    while(!itK.IsAtEnd() ) {
      itI.Set(itI.Get() * itK.Get() );
      ++itI;
      ++itK;
      }
    }

  //Inverse FFT image
#if ITK_VERSION_MAJOR <= 3
  typedef itk::FFTComplexConjugateToRealImageFilter< FFTPrecisionType, ImageDimension > IFFTType;
#else
  typedef itk::FFTComplexConjugateToRealImageFilter< typename FFTType::OutputImageType > IFFTType;
#endif
  typename IFFTType::Pointer ifft = IFFTType::New();
  ifft->SetInput( fftI->GetOutput() );
  ifft->SetActualXDimensionIsOdd( paddedImage->GetLargestPossibleRegion().GetSize(0) % 2 );
  ifft->SetNumberOfThreads( 1 );
  ifft->SetReleaseDataFlag( true );
  ifft->Update();

  // Crop and paste result
  itk::ImageRegionConstIterator<FFTInputImageType> itS(ifft->GetOutput(), outputRegionForThread);
  itk::ImageRegionIterator<OutputImageType>        itD(this->GetOutput(), outputRegionForThread);
  itS.GoToBegin();
  itD.GoToBegin();
  while(!itS.IsAtEnd() ) {
    itD.Set(itS.Get() );
    ++itS;
    ++itD;
    }
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
typename FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>::FFTInputImagePointer
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::PadInputImageRegion(const RegionType &inputRegion)
{
  UpdateTruncationMirrorWeights();

  RegionType paddedRegion = inputRegion;

  // Set x padding
  typename SizeType::SizeValueType xPaddedSize = 2*inputRegion.GetSize(0);
  while( GreatestPrimeFactor( xPaddedSize ) > m_GreatestPrimeFactor )
    xPaddedSize++;
  paddedRegion.SetSize(0, xPaddedSize);
  long zeroext = ( (long)xPaddedSize - (long)inputRegion.GetSize(0) ) / 2;
  paddedRegion.SetIndex(0, inputRegion.GetIndex(0) - zeroext);

  // Set y padding
  typename SizeType::SizeValueType yPaddedSize = inputRegion.GetSize(1);
  while( GreatestPrimeFactor( yPaddedSize ) > m_GreatestPrimeFactor )
    yPaddedSize++;
  paddedRegion.SetSize(1, yPaddedSize);
  paddedRegion.SetIndex(1, inputRegion.GetIndex(1) );

  // Create padded image (spacing and origin do not matter)
  FFTInputImagePointer paddedImage = FFTInputImageType::New();
  paddedImage->SetRegions(paddedRegion);
  paddedImage->Allocate();
  paddedImage->FillBuffer(0);

  const long next = vnl_math_min(zeroext, (long)this->GetTruncationCorrectionExtent() );
  if(next)
    {
    typename FFTInputImageType::IndexType idx;
    typename FFTInputImageType::IndexType::IndexValueType borderDist=0, rightIdx=0;

    // Mirror left
    RegionType leftRegion = inputRegion;
    leftRegion.SetIndex(0, inputRegion.GetIndex(0)-next);
    leftRegion.SetSize(0, next);
    itk::ImageRegionIteratorWithIndex<FFTInputImageType> itLeft(paddedImage, leftRegion);
    while(!itLeft.IsAtEnd() )
      {
      idx = itLeft.GetIndex();
      borderDist = inputRegion.GetIndex(0)-idx[0];
      idx[0] = inputRegion.GetIndex(0) + borderDist;
      itLeft.Set(m_TruncationMirrorWeights[ borderDist ] * this->GetInput()->GetPixel(idx) );
      ++itLeft;
      }

    // Mirror right
    RegionType rightRegion = inputRegion;
    rightRegion.SetIndex(0, inputRegion.GetIndex(0)+inputRegion.GetSize(0) );
    rightRegion.SetSize(0, next);
    itk::ImageRegionIteratorWithIndex<FFTInputImageType> itRight(paddedImage, rightRegion);
    while(!itRight.IsAtEnd() )
      {
      idx = itRight.GetIndex();
      rightIdx = inputRegion.GetIndex(0)+inputRegion.GetSize(0)-1;
      borderDist = idx[0]-rightIdx;
      idx[0] = rightIdx - borderDist;
      itRight.Set(m_TruncationMirrorWeights[ borderDist ] * this->GetInput()->GetPixel(idx) );
      ++itRight;
      }
    }

  // Copy central part
  itk::ImageRegionConstIterator<InputImageType> itS(this->GetInput(), inputRegion);
  itk::ImageRegionIterator<FFTInputImageType>   itD(paddedImage, inputRegion);
  itS.GoToBegin();
  itD.GoToBegin();
  while(!itS.IsAtEnd() ) {
    itD.Set(itS.Get() );
    ++itS;
    ++itD;
    }

  return paddedImage;
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::PrintSelf(std::ostream &os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "GreatestPrimeFactor: "  << m_GreatestPrimeFactor << std::endl;
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
bool
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::IsPrime( int n ) const
{
  int last = (int)vcl_sqrt( double(n) );

  for( int x=2; x<=last; x++ )
    if( n%x == 0 )
      return false;
  return true;
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
int
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::GreatestPrimeFactor( int n ) const
{
  int v = 2;

  while( v <= n )
    if( n%v == 0 && IsPrime( v ) )
      n /= v;
    else
      v += 1;
  return v;
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
typename FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>::FFTOutputImagePointer
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::GetFFTRampKernel(const int width)
{
  // Allocate kernel
  FFTInputImagePointer kernel = FFTInputImageType::New();
  SizeType             size;

  size.Fill(1);
  size[0] = width;
  kernel->SetRegions( size );
  kernel->Allocate();
  kernel->FillBuffer(0.);

  // Compute kernel in space domain (see Kak & Slaney, chapter 3 equation 61
  // page 72) although spacing is not squared according to equation 69 page 75
  double spacing = this->GetInput()->GetSpacing()[0];
  IndexType i,j;
  i.Fill(0);
  j.Fill(0);
  kernel->SetPixel(i, 1./(4.*spacing) );
  for(i[0]=1, j[0]=size[0]-1; i[0] < typename IndexType::IndexValueType(size[0]/2); i[0]+=2, j[0]-=2) {
    double v = i[0] * vnl_math::pi;
    v = -1. / (v * v * spacing);
    kernel->SetPixel(i, v);
    kernel->SetPixel(j, v);
    }

  // FFT kernel
#if ITK_VERSION_MAJOR <= 3
  typedef itk::FFTRealToComplexConjugateImageFilter< FFTPrecisionType, ImageDimension > FFTType;
#else
  typedef itk::FFTRealToComplexConjugateImageFilter< FFTInputImageType > FFTType;
#endif
  typename FFTType::Pointer fftK = FFTType::New();
  fftK->SetInput( kernel );
  fftK->SetNumberOfThreads( 1 );
  fftK->Update();

  // Windowing (if enabled)
  typedef itk::ImageRegionIterator<typename FFTType::OutputImageType> IteratorType;
  IteratorType itK(fftK->GetOutput(), fftK->GetOutput()->GetLargestPossibleRegion() );

  unsigned int n = fftK->GetOutput()->GetLargestPossibleRegion().GetSize(0);
  n = itk::Math::Round<double>(n * vnl_math_min(1.0, this->GetHannCutFrequency() ) );

  itK.GoToBegin();
  if(this->GetHannCutFrequency()>0.)
    {
    for(unsigned int i=0; i<n; i++, ++itK)
      itK.Set( itK.Get() * TFFTPrecision(0.5*(1+vcl_cos(vnl_math::pi*i/n))));
    }
  else if(this->GetCosineCutFrequency() > 0.)
    {
    for(unsigned int i=0; i<n; i++, ++itK)
      itK.Set( itK.Get() * TFFTPrecision(vcl_cos(0.5*vnl_math::pi*i/n)));
    }
  else if(this->GetHammingFrequency() > 0.)
    {
    for(unsigned int i=0; i<n; i++, ++itK)
      itK.Set( itK.Get() * TFFTPrecision(0.54+0.46*(vcl_cos(vnl_math::pi*i/n))));
    }
  else
    itK.GoToEnd();

  for( ; !itK.IsAtEnd(); ++itK)
    itK.Set( itK.Get() * TFFTPrecision(0.) );

  return fftK->GetOutput();
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::UpdateTruncationMirrorWeights()
{
  const unsigned int next = this->GetTruncationCorrectionExtent();

  if ( (unsigned int) m_TruncationMirrorWeights.size() != next)
    {
    m_TruncationMirrorWeights.resize(next+1);
    for(unsigned int i=0; i<next+1; i++)
      m_TruncationMirrorWeights[i] = pow( sin( (next-i)*vnl_math::pi/(2*next-2) ), 0.75);
    }
}

} // end namespace rtk
#endif
