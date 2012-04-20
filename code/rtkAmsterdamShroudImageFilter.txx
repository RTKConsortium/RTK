#ifndef __rtkAmsterdamShroudImageFilter_txx
#define __rtkAmsterdamShroudImageFilter_txx

#include <itkImageFileWriter.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
AmsterdamShroudImageFilter<TInputImage, TOutputImage>
::AmsterdamShroudImageFilter()
{
  m_DerivativeFilter = DerivativeType::New();
  m_NegativeFilter = NegativeType::New();
  m_ThresholdFilter = ThresholdType::New();
  m_SumFilter = SumType::New();
  m_ConvolutionFilter = ConvolutionType::New();
  m_SubtractFilter = SubtractType::New();
  m_PermuteFilter = PermuteType::New();

  m_NegativeFilter->SetInput( m_DerivativeFilter->GetOutput() );
  m_ThresholdFilter->SetInput( m_NegativeFilter->GetOutput() );
  m_SumFilter->SetInput( m_ThresholdFilter->GetOutput() );
  m_ConvolutionFilter->SetInput( m_SumFilter->GetOutput() );
  m_SubtractFilter->SetInput1( m_SumFilter->GetOutput() );
  m_SubtractFilter->SetInput2( m_ConvolutionFilter->GetOutput() );
  m_PermuteFilter->SetInput( m_SubtractFilter->GetOutput() );

  m_DerivativeFilter->SetOrder(DerivativeType::FirstOrder);
  m_DerivativeFilter->SetDirection(1);
  m_DerivativeFilter->SetSigma(4);

  m_NegativeFilter->SetConstant(-1.0);
  m_NegativeFilter->InPlaceOn();

  m_ThresholdFilter->SetUpper(0.0);
  m_ThresholdFilter->SetOutsideValue(0.0);
  m_ThresholdFilter->InPlaceOn();

  m_SumFilter->SetProjectionDimension(0);

  // Unsharp mask: difference between image and moving average
  // m_ConvolutionFilter computes the moving average
  typename TOutputImage::Pointer kernel = TOutputImage::New();
  typename TOutputImage::RegionType region;
  region.SetIndex(0, 0);
  region.SetIndex(1, -8);
  region.SetSize(0, 1);
  region.SetSize(1, 17);
  kernel->SetRegions(region);
  kernel->Allocate();
  kernel->FillBuffer(1.);
  m_ConvolutionFilter->SetImageKernelInput( kernel );

  // The permute filter is used to allow streaming because ITK splits the output image in the last direction
  typename PermuteType::PermuteOrderArrayType order;
  order[0] = 1;
  order[1] = 0;
  m_PermuteFilter->SetOrder(order);
}

template <class TInputImage, class TOutputImage>
void
AmsterdamShroudImageFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{
  // get pointers to the input and output
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput();

  if ( !outputPtr || !inputPtr)
    {
    return;
    }

  m_DerivativeFilter->SetInput( this->GetInput() );
  m_PermuteFilter->UpdateOutputInformation();
  outputPtr->SetLargestPossibleRegion( m_PermuteFilter->GetOutput()->GetLargestPossibleRegion() );
}

template <class TInputImage, class TOutputImage>
void
AmsterdamShroudImageFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer  inputPtr =
    const_cast< TInputImage * >( this->GetInput() );
  if ( !inputPtr )
    {
    return;
    }
  m_DerivativeFilter->SetInput( this->GetInput() );
  m_PermuteFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion() );
  m_PermuteFilter->GetOutput()->PropagateRequestedRegion();
}

template<class TInputImage, class TOutputImage>
void
AmsterdamShroudImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  m_PermuteFilter->Update();
  this->GraftOutput( m_PermuteFilter->GetOutput() );
}

} // end namespace rtk
#endif
