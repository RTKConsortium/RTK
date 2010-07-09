#ifndef __itkFFTRampImageFilter_txx
#define __itkFFTRampImageFilter_txx

#include <itkRegionOfInterestImageFilter.h>
#include <itkConstantPadImageFilter.h>
#include <itkFFTRealToComplexConjugateImageFilter.h>
#include <itkFFTComplexConjugateToRealImageFilter.h>
#include <itkCropImageFilter.h>
#include <itkPasteImageFilter.h>

namespace itk
{

template <class TInputImage, class TOutputImage, class TFFTPrecision>
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::FFTRampImageFilter()
{
  m_GreatestPrimeFactor = 2;
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
void
FFTRampImageFilter<TInputImage, TOutputImage, TFFTPrecision>
::GenerateData()
{
  this->AllocateOutputs();

  // ROI of the image that is processed
  typedef itk::RegionOfInterestImageFilter< InputImageType, InputImageType > RegionOfInterestType;
  typename RegionOfInterestType::Pointer roi = RegionOfInterestType::New();
  roi->SetInput( this->GetInput() );
  roi->SetNumberOfThreads( this->GetNumberOfThreads() );
  roi->SetReleaseDataFlag( true );
  roi->SetRegionOfInterest( this->GetOutput()->GetRequestedRegion() );
  roi->Update();

  // Compute padding parameters
  SizeType padSize, inputSize, padLowerBound, padUpperBound;
  inputSize = this->GetInput()->GetLargestPossibleRegion().GetSize();
  padSize = inputSize;
  padSize[0] *= 2;
  while( GreatestPrimeFactor( padSize[0] ) > m_GreatestPrimeFactor )
    padSize[0]++;
  padLowerBound.Fill(0);
  padLowerBound[0] = ( padSize[0] - inputSize[0] ) / 2;
  padUpperBound =   padSize - inputSize - padLowerBound;

  // Pad image
  typedef itk::ConstantPadImageFilter< InputImageType, InputImageType > ConstantPadType;
  typename ConstantPadType::Pointer pad = ConstantPadType::New();
  pad->SetInput( roi->GetOutput() );
  pad->SetNumberOfThreads( this->GetNumberOfThreads() );
  pad->SetReleaseDataFlag( true );
  pad->SetPadLowerBound( padLowerBound );
  pad->SetPadUpperBound( padUpperBound );
  pad->Update();

  // FFT padded image
  typedef itk::FFTRealToComplexConjugateImageFilter< FFTPrecisionType, ImageDimension > FFTType;
  typename FFTType::Pointer fftI = FFTType::New();
  fftI->SetInput( pad->GetOutput() );
  fftI->SetNumberOfThreads( this->GetNumberOfThreads() );
  fftI->SetReleaseDataFlag( true );
  fftI->Update();

  // Allocate kernel
  InputImagePointer kernel = InputImageType::New();
  SizeType size;
  size.Fill(1);
  size[0] = padSize[0];
  kernel->SetRegions( size );
  kernel->Allocate();
  kernel->FillBuffer(0.);

  // Compute kernel in space domain
  // (see Kak & Slaney, chapter 3 page 91 equation 124)
  const double spacing = this->GetInput()->GetSpacing()[0];
  IndexType i,j;
  i.Fill(0);
  j.Fill(0);
  kernel->SetPixel(i, 1./(4.*spacing*spacing));
  for(i[0]=1, j[0]=size[0]-1; i[0] < typename IndexType::IndexValueType(size[0]/2); i[0]+=2, j[0]-=2) {
    double v = i[0] * vnl_math::pi * spacing;
    v = -1. / (v*v);
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
  ifft->SetActualXDimensionIsOdd( pad->GetOutput()->GetLargestPossibleRegion().GetSize()[0] % 2 );
  ifft->SetNumberOfThreads( this->GetNumberOfThreads() );
  ifft->SetReleaseDataFlag( true );
  ifft->Update();

  //Crop and paste result (combination of itk::CropImageFilter and itk::PasteImageFilter, but the
  //latter is not working properly for a stream)
  typedef itk::ImageRegionIterator<OutputImageType> OutputIteratorType;
  OutputIteratorType itS(ifft->GetOutput(), roi->GetOutput()->GetLargestPossibleRegion());
  OutputIteratorType itD(this->GetOutput(), this->GetOutput()->GetRequestedRegion());
  itS.GoToBegin();
  itD.GoToBegin();
  while(!itS.IsAtEnd()) {
    itD.Set(itS.Get());
    ++itS;
    ++itD;
  }
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
