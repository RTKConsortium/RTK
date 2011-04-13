#ifndef __itkAmsterdamShroudImageFilter_txx
#define __itkAmsterdamShroudImageFilter_txx

#include <itkImageFileWriter.h>

namespace itk
{

template <class TInputImage, class TOutputImage>
AmsterdamShroudImageFilter<TInputImage, TOutputImage>
::AmsterdamShroudImageFilter()
{
  m_DerivativeFilter = DerivativeType::New();
  m_NegativeFilter = NegativeType::New();
  m_ThresholdFilter = ThresholdType::New();
  m_SumFilter = SumType::New();
  m_SmoothFilter = SmoothType::New();
  m_SubtractFilter = SubtractType::New();

  m_NegativeFilter->SetInput( m_DerivativeFilter->GetOutput() );
  m_ThresholdFilter->SetInput( m_NegativeFilter->GetOutput() );
  m_SumFilter->SetInput( m_ThresholdFilter->GetOutput() );
  m_SmoothFilter->SetInput( m_SumFilter->GetOutput() );
  m_SubtractFilter->SetInput1( m_SumFilter->GetOutput() );
  m_SubtractFilter->SetInput2( m_SmoothFilter->GetOutput() );

  m_DerivativeFilter->SetOrder(DerivativeType::FirstOrder);
  m_DerivativeFilter->SetDirection(1);
  m_DerivativeFilter->SetSigma(4);
  m_NegativeFilter->SetConstant(-1.0);
  m_ThresholdFilter->SetUpper(0.0);
  m_ThresholdFilter->SetOutsideValue(0.0);
  m_SumFilter->SetProjectionDimension(0);
  m_SmoothFilter->SetSigma(40);
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
  m_SumFilter->UpdateOutputInformation();
  outputPtr->SetLargestPossibleRegion( m_SumFilter->GetOutput()->GetLargestPossibleRegion() );
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
  inputPtr->SetRequestedRegion( this->GetInput()->GetLargestPossibleRegion() );
}

template<class TInputImage, class TOutputImage>
void
AmsterdamShroudImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  this->AllocateOutputs();
  m_SubtractFilter->Update();
  this->GraftOutput( m_SubtractFilter->GetOutput() );
}

} // end namespace itk
#endif
