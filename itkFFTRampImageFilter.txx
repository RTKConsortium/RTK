#ifndef __itkFFTRampImageFilter_txx
#define __itkFFTRampImageFilter_txx

#include <itkFFTRealToComplexConjugateImageFilter.h>
#include <itkFFTComplexConjugateToRealImageFilter.h>
#include <itkImageFileWriter.h>

namespace itk
{

template <class TInputImage, class TOutputImage, class TFFTPrecision>
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::FFTRampImageFilter():m_TruncationCorrection(0.), m_GreatestPrimeFactor(2)
{
  this->SetNumberOfThreads(1);
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  InputImageType * input = const_cast<InputImageType *>(this->GetInput());
  if ( !input )
    return;

  // Compute input region (==requested region fully enlarged for dim 0)
  RegionType inputRegion;
  this->CallCopyOutputRegionToInputRegion(inputRegion, this->GetOutput()->GetRequestedRegion());
  input->SetRequestedRegion( inputRegion );
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
int
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::GetTruncationCorrectionExtent()
{
  return Math::Floor<TFFTPrecision>(m_TruncationCorrection * this->GetInput()->GetRequestedRegion().GetSize(0));
}


template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::BeforeThreadedGenerateData()
{
  // Pre compute weights for truncation correction in a lookup table. The index
  // is the distance to the original image border.
  const long next = this->GetTruncationCorrectionExtent();
  if ((long)m_TruncationMirrorWeights.size() != next)
    {
    m_TruncationMirrorWeights.resize(next);
    for(unsigned int i=0; i<next; i++)
      m_TruncationMirrorWeights[i] = pow( sin((next-i)*vnl_math::pi/(2*next-2)), 0.75);
    }
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::ThreadedGenerateData( const RegionType& outputRegionForThread, int threadId )
{
  // Pad image region
  FFTImagePointer paddedImage = PadInputImageRegion(outputRegionForThread);

  // FFT padded image
  typedef itk::FFTRealToComplexConjugateImageFilter< FFTPrecisionType, ImageDimension > FFTType;
  typename FFTType::Pointer fftI = FFTType::New();
  fftI->SetInput( paddedImage );
  fftI->SetNumberOfThreads( this->GetNumberOfThreads() );
  fftI->SetReleaseDataFlag( true );
  fftI->Update();

  // Allocate kernel
  FFTImagePointer kernel = FFTImageType::New();
  SizeType size;
  size.Fill(1);
  size[0] = paddedImage->GetLargestPossibleRegion().GetSize(0);
  kernel->SetRegions( size );
  kernel->Allocate();
  kernel->FillBuffer(0.);

  // Compute kernel in space domain (see Kak & Slaney, chapter 3 equation 61 page 72)
  // although spacing is not squared according to equation 69 page 75
  const double spacing = this->GetInput()->GetSpacing()[0];
  IndexType i,j;
  i.Fill(0);
  j.Fill(0);
  kernel->SetPixel(i, 1./(4.*spacing));
  for(i[0]=1, j[0]=size[0]-1; i[0] < typename IndexType::IndexValueType(size[0]/2); i[0]+=2, j[0]-=2) {
    double v = i[0] * vnl_math::pi;
    v = -1. / (v * v * spacing);
    kernel->SetPixel(i, v);
    kernel->SetPixel(j, v);
  }

  // FFT kernel
  typedef itk::FFTRealToComplexConjugateImageFilter< FFTPrecisionType, ImageDimension > FFTType;
  typename FFTType::Pointer fftK = FFTType::New();
  fftK->SetInput( kernel );
  fftK->SetNumberOfThreads( this->GetNumberOfThreads() );
  fftK->SetReleaseDataFlag( false );
  fftK->Update();

  //Multiply line-by-line
  typedef itk::ImageRegionIterator<typename FFTType::TOutputImageType> IteratorType;
  IteratorType itI(fftI->GetOutput(), fftI->GetOutput()->GetLargestPossibleRegion());
  IteratorType itK(fftK->GetOutput(), fftK->GetOutput()->GetLargestPossibleRegion());
  itI.GoToBegin();
  while(!itI.IsAtEnd()) {
    itK.GoToBegin();
    while(!itK.IsAtEnd()) {
      itI.Set(itI.Get() * itK.Get());
      ++itI;
      ++itK;
    }
  }

  //Inverse FFT image
  typedef itk::FFTComplexConjugateToRealImageFilter< FFTPrecisionType, ImageDimension > IFFTType;
  typename IFFTType::Pointer ifft = IFFTType::New();
  ifft->SetInput( fftI->GetOutput() );
  ifft->SetActualXDimensionIsOdd( paddedImage->GetLargestPossibleRegion().GetSize(0) % 2 );
  ifft->SetNumberOfThreads( this->GetNumberOfThreads() );
  ifft->SetReleaseDataFlag( true );
  ifft->Update();

  //Crop and paste result (combination of itk::CropImageFilter and itk::PasteImageFilter, but the
  //latter is not working properly for a stream)
  itk::ImageRegionConstIterator<FFTImageType> itS(ifft->GetOutput(), outputRegionForThread);
  itk::ImageRegionIterator<OutputImageType> itD(this->GetOutput(), outputRegionForThread);
  itS.GoToBegin();
  itD.GoToBegin();
  while(!itS.IsAtEnd()) {
    itD.Set(itS.Get());
    ++itS;
    ++itD;
  }
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
typename FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>::FFTImagePointer
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::PadInputImageRegion(const RegionType &inputRegion)
{
  typename SizeType::SizeValueType xPaddedSize = 2*inputRegion.GetSize(0);
  while( GreatestPrimeFactor( xPaddedSize ) > m_GreatestPrimeFactor )
    xPaddedSize++;

  RegionType paddedRegion = inputRegion;
  paddedRegion.SetSize(0, xPaddedSize);
  paddedRegion.SetIndex(0, inputRegion.GetIndex(0)+((long)inputRegion.GetSize(0)-(long)xPaddedSize) / 2);

  // Create padded image (spacing and origin do not matter)
  FFTImagePointer paddedImage = FFTImageType::New();
  paddedImage->SetRegions(paddedRegion);
  paddedImage->Allocate();
  paddedImage->FillBuffer(0);

  const long next = this->GetTruncationCorrectionExtent();
  if(next)
    {

    // Mirror left
    RegionType leftRegion = paddedRegion;
    leftRegion.SetIndex(0, inputRegion.GetIndex(0)-next+1);
    leftRegion.SetSize(0, next);
    itk::ImageRegionIteratorWithIndex<FFTImageType> itLeft(paddedImage, leftRegion);
    while(!itLeft.IsAtEnd())
      {
      typename FFTImageType::IndexType idx = itLeft.GetIndex();
      idx[0] *= -1;
      itLeft.Set(m_TruncationMirrorWeights[ idx[0] ] * this->GetInput()->GetPixel(idx));
      ++itLeft;
      }

    // Mirror right
    RegionType rightRegion = paddedRegion;
    rightRegion.SetIndex(0, inputRegion.GetIndex(0)+inputRegion.GetSize(0));
    rightRegion.SetSize(0, next);
    itk::ImageRegionIteratorWithIndex<FFTImageType> itRight(paddedImage, rightRegion);
    while(!itRight.IsAtEnd())
      {
      typename FFTImageType::IndexType idx = itRight.GetIndex();
      typename FFTImageType::IndexType::IndexValueType borderDist = idx[0]-inputRegion.GetSize(0);
      idx[0] = inputRegion.GetSize(0)-borderDist-2;
      itRight.Set(m_TruncationMirrorWeights[ borderDist ] * this->GetInput()->GetPixel(idx));
      ++itRight;
      }
    }

  // Copy central part
  itk::ImageRegionConstIterator<InputImageType> itS(this->GetInput(), inputRegion);
  itk::ImageRegionIterator<FFTImageType>   itD(paddedImage, inputRegion);
  itS.GoToBegin();
  itD.GoToBegin();
  while(!itS.IsAtEnd()) {
    itD.Set(itS.Get());
    ++itS;
    ++itD;
  }

  return paddedImage;
}

template<class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::PrintSelf(std::ostream &os, Indent indent) const
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

} // end namespace itk
#endif
