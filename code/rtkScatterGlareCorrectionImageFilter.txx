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

#ifndef __rtkScatterGlareCorrectionImageFilter_txx
#define __rtkScatterGlareCorrectionImageFilter_txx

// Use local RTK FFTW files taken from Gaëtan Lehmann's code for
// thread safety: http://hdl.handle.net/10380/3154
#include <itkRealToHalfHermitianForwardFFTImageFilter.h>
#include <itkHalfHermitianToRealInverseFFTImageFilter.h>

#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

namespace rtk
{

template <class TInputImage, class TOutputImage, class TFFTPrecision>
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::ScatterGlareCorrectionImageFilter() :
  m_TruncationCorrection(0.), m_GreatestPrimeFactor(2), m_HannCutFrequency(0.),
  m_HannCutFrequencyY(0.), m_BackupNumberOfThreads(1)
{
#if defined(USE_FFTWD)
  if(typeid(TFFTPrecision).name() == typeid(double).name() )
    m_GreatestPrimeFactor = 13;
#endif
#if defined(USE_FFTWF)
  if(typeid(TFFTPrecision).name() == typeid(float).name() )
    m_GreatestPrimeFactor = 13;
#endif
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
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
  inputRegion.SetIndex(0, this->GetOutput()->GetLargestPossibleRegion().GetIndex(0) );
  inputRegion.SetSize(0, this->GetOutput()->GetLargestPossibleRegion().GetSize(0) );

  // Also enlarge along dim 1 if hann is set in that direction
  if(m_HannCutFrequencyY>0.)
    {
    inputRegion.SetIndex(1, this->GetOutput()->GetLargestPossibleRegion().GetIndex(1) );
    inputRegion.SetSize(1, this->GetOutput()->GetLargestPossibleRegion().GetSize(1) );
    }
  input->SetRequestedRegion( inputRegion );
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
int
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::GetTruncationCorrectionExtent()
{
  return vnl_math_floor(m_TruncationCorrection * this->GetInput()->GetRequestedRegion().GetSize(0));
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::BeforeThreadedGenerateData()
{
  UpdateTruncationMirrorWeights();

  // If the following condition is met, multi-threading is left to the (i)fft
  // filter. Otherwise, one splits the image and a separate fft is performed
  // per thread.
  if(this->GetOutput()->GetRequestedRegion().GetSize()[2] == 1 &&
     this->GetHannCutFrequencyY() != 0.)
    {
    m_BackupNumberOfThreads = this->GetNumberOfThreads();
    this->SetNumberOfThreads(1);
    }
  else
    m_BackupNumberOfThreads = 1;

#if !(ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 3))
  if (this->GetNumberOfThreads() > 1)
    {
    itkWarningMacro(<< "ITK versions before 4.3 have a multithreading issue in FFTW, upgrade ITK for better performances."
                    << "See http://www.itk.org/gitweb?p=ITK.git;a=commit;h=a0661da4252fcdd638c6415c89cd2f26edd9f553 for more information.");
    this->SetNumberOfThreads(1);
    }
#endif
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::AfterThreadedGenerateData()
{
  if(this->GetOutput()->GetRequestedRegion().GetSize()[2] == 1 &&
     this->GetHannCutFrequencyY() != 0.)
    this->SetNumberOfThreads(m_BackupNumberOfThreads);
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::ThreadedGenerateData( const RegionType& outputRegionForThread, ThreadIdType itkNotUsed(threadId) )
{
  // Pad image region enlarged along X
  RegionType enlargedRegionX = outputRegionForThread;
  enlargedRegionX.SetIndex(0, this->GetInput()->GetRequestedRegion().GetIndex(0) );
  enlargedRegionX.SetSize(0, this->GetInput()->GetRequestedRegion().GetSize(0) );
  enlargedRegionX.SetIndex(1, this->GetInput()->GetRequestedRegion().GetIndex(1) );
  enlargedRegionX.SetSize(1, this->GetInput()->GetRequestedRegion().GetSize(1) );
  FFTInputImagePointer paddedImage;
  paddedImage = PadInputImageRegion(enlargedRegionX);

  // FFT padded image
  typedef itk::RealToHalfHermitianForwardFFTImageFilter< FFTInputImageType > FFTType;
  typename FFTType::Pointer fftI = FFTType::New();
  fftI->SetInput( paddedImage );
  fftI->SetNumberOfThreads( m_BackupNumberOfThreads );
  fftI->Update();

  // Get FFT ramp kernel
  typename FFTOutputImageType::SizeType s;
  s = paddedImage->GetLargestPossibleRegion().GetSize();
  FFTOutputImagePointer fftK = this->GetFFTRampKernel(s[0], s[1]);

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
  typedef itk::HalfHermitianToRealInverseFFTImageFilter< typename FFTType::OutputImageType > IFFTType;
  typename IFFTType::Pointer ifft = IFFTType::New();
  ifft->SetInput( fftI->GetOutput() );
  ifft->SetNumberOfThreads( m_BackupNumberOfThreads );
  ifft->SetReleaseDataFlag( true );
  ifft->Update();

  // Crop and paste result
  itk::ImageRegionConstIterator<FFTInputImageType> itS(ifft->GetOutput(), outputRegionForThread);
  itk::ImageRegionIterator<OutputImageType>        itD(this->GetOutput(), outputRegionForThread);
  itS.GoToBegin();
  itD.GoToBegin();
  while(!itS.IsAtEnd() )
    {
    itD.Set(itS.Get() );
    ++itS;
    ++itD;
    }
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
typename ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::FFTInputImagePointer
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::PadInputImageRegion(const RegionType &inputRegion)
{
  UpdateTruncationMirrorWeights();
  RegionType paddedRegion = GetPaddedImageRegion(inputRegion);

  // Create padded image (spacing and origin do not matter)
  FFTInputImagePointer paddedImage = FFTInputImageType::New();
  paddedImage->SetRegions(paddedRegion);
  paddedImage->Allocate();
  paddedImage->FillBuffer(0);

  const long next = vnl_math_min(inputRegion.GetIndex(0) - paddedRegion.GetIndex(0),
                                 (typename FFTInputImageType::IndexValueType)this->GetTruncationCorrectionExtent() );
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
typename ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::RegionType
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::GetPaddedImageRegion(const RegionType &inputRegion)
{
  RegionType paddedRegion = inputRegion;

  // Set x padding
  typename SizeType::SizeValueType xPaddedSize = 2*inputRegion.GetSize(0);
  while( GreatestPrimeFactor( xPaddedSize ) > m_GreatestPrimeFactor )
    xPaddedSize++;
  paddedRegion.SetSize(0, xPaddedSize);
  long zeroext = ( (long)xPaddedSize - (long)inputRegion.GetSize(0) ) / 2;
  paddedRegion.SetIndex(0, inputRegion.GetIndex(0) - zeroext);

  // Set y padding. Padding along Y is only required if
  // - there is some windowing in the Y direction
  // - the DFT requires the size to be the product of given prime factors
  typename SizeType::SizeValueType yPaddedSize = inputRegion.GetSize(1);
  if(this->GetHannCutFrequencyY()>0.)
    yPaddedSize *= 2;
  while( GreatestPrimeFactor( yPaddedSize ) > m_GreatestPrimeFactor )
    yPaddedSize++;
  paddedRegion.SetSize(1, yPaddedSize);
  paddedRegion.SetIndex(1, inputRegion.GetIndex(1) );

  return paddedRegion;
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::PrintSelf(std::ostream &os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "GreatestPrimeFactor: "  << m_GreatestPrimeFactor << std::endl;
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
bool
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
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
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
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
typename ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::FFTOutputImagePointer
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::GetFFTRampKernel(const int width, const int height)
{
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
    double v = ix[0] * vnl_math::pi;
    v = -1. / (v * v * spacing);
    kernel->SetPixel(ix, v);
    kernel->SetPixel(jx, v);
    }

  // FFT kernel
  typedef itk::RealToHalfHermitianForwardFFTImageFilter< FFTInputImageType, FFTOutputImageType > FFTType;
  typename FFTType::Pointer fftK = FFTType::New();
  fftK->SetInput( kernel );
  fftK->SetNumberOfThreads( 1 );
  fftK->Update();

  // Windowing (if enabled)
  typedef itk::ImageRegionIteratorWithIndex<typename FFTType::OutputImageType> IteratorType;
  IteratorType itK(fftK->GetOutput(), fftK->GetOutput()->GetLargestPossibleRegion() );

  unsigned int n = fftK->GetOutput()->GetLargestPossibleRegion().GetSize(0);

  itK.GoToBegin();
  const unsigned int ncut = itk::Math::Round<double>(n * vnl_math_min(1.0, this->GetHannCutFrequency() ) );
  for(unsigned int i=0; i<ncut; i++, ++itK)
    itK.Set( itK.Get() * TFFTPrecision(0.5*(1+vcl_cos(vnl_math::pi*i/ncut))));
 
  for(; !itK.IsAtEnd(); ++itK)
    {
    itK.Set( itK.Get() * TFFTPrecision(0.) );
    }

  // Replicate and window if required
  FFTOutputImagePointer result = fftK->GetOutput();
  if(this->GetHannCutFrequencyY()>0.)
    {
    size.Fill(1);
    size[0] = fftK->GetOutput()->GetLargestPossibleRegion().GetSize(0);
    size[1] = height;

    const unsigned int ncut = itk::Math::Round<double>( (height/2+1) * vnl_math_min(1.0, this->GetHannCutFrequencyY() ) );

    result = FFTOutputImageType::New();
    result->SetRegions( size );
    result->Allocate();
    result->FillBuffer(0.);

    IteratorType itTwoDK(result, result->GetLargestPossibleRegion() );
    for(unsigned int j=0; j<ncut; j++)
      {
      itK.GoToBegin();
      const TFFTPrecision win( 0.5*( 1+vcl_cos(vnl_math::pi*j/ncut) ) );
      for(unsigned int i=0; i<size[0]; ++itK, ++itTwoDK, i++)
        {
        itTwoDK.Set( win * itK.Get() );
        }
      }
    itTwoDK.GoToReverseBegin();
    for(unsigned int j=1; j<ncut; j++)
      {
      itK.GoToReverseBegin();
      const TFFTPrecision win( 0.5*( 1+vcl_cos(vnl_math::pi*j/ncut) ) );
      for(unsigned int i=0; i<size[0]; --itK, --itTwoDK, i++)
        {
        itTwoDK.Set( win * itK.Get() );
        }
      }
    }

  return result;
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>
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
