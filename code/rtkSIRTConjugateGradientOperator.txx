#ifndef __rtkSIRTConjugateGradientOperator_txx
#define __rtkSIRTConjugateGradientOperator_txx

#include "rtkSIRTConjugateGradientOperator.h"

namespace rtk
{

template< typename TOutputImage>
SIRTConjugateGradientOperator<TOutputImage>::SIRTConjugateGradientOperator()
{
  this->SetNumberOfRequiredInputs(2);
  m_MultiplyFilter = MultiplyFilterType::New();
  m_ZeroMultiplyProjectionFilter = MultiplyFilterType::New();
  m_ZeroMultiplyVolumeFilter = MultiplyFilterType::New();

  // Set permanent parameters
  m_ZeroMultiplyProjectionFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_ZeroMultiplyVolumeFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());

  // Set memory management options
  m_ZeroMultiplyProjectionFilter->ReleaseDataFlagOn();
  m_ZeroMultiplyVolumeFilter->ReleaseDataFlagOn();
  m_MultiplyFilter->ReleaseDataFlagOn();
}

template< typename TOutputImage >
void
SIRTConjugateGradientOperator<TOutputImage>
::SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg)
{
  m_BackProjectionFilter = _arg;
}

template< typename TOutputImage >
void
SIRTConjugateGradientOperator<TOutputImage>
::SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg)
{
  m_ForwardProjectionFilter = _arg;
}


template< typename TOutputImage >
void
SIRTConjugateGradientOperator<TOutputImage>
::SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg)
{
  m_BackProjectionFilter->SetGeometry(_arg.GetPointer());
  m_ForwardProjectionFilter->SetGeometry(_arg);
}

template< typename TOutputImage >
void
SIRTConjugateGradientOperator<TOutputImage>
::GenerateInputRequestedRegion()
{
  // Input 0 is the volume in which we backproject
  typename Superclass::InputImagePointer inputPtr0 =
    const_cast< TOutputImage * >( this->GetInput(0) );
  if ( !inputPtr0 ) return;
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the stack of projections to backproject
  typename Superclass::InputImagePointer  inputPtr1 =
    const_cast< TOutputImage * >( this->GetInput(1) );
  if ( !inputPtr1 ) return;
  inputPtr1->SetRequestedRegion( inputPtr1->GetLargestPossibleRegion() );
}

template< typename TOutputImage >
void
SIRTConjugateGradientOperator<TOutputImage>
::GenerateOutputInformation()
{
  // Set runtime connections, and connections with
  // forward and back projection filters, which are set
  // at runtime
  m_ForwardProjectionFilter->SetInput(0, m_ZeroMultiplyProjectionFilter->GetOutput());
  m_BackProjectionFilter->SetInput(0, m_ZeroMultiplyVolumeFilter->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_ForwardProjectionFilter->GetOutput());
  m_ZeroMultiplyVolumeFilter->SetInput1(this->GetInput(0));
  m_ZeroMultiplyProjectionFilter->SetInput1(this->GetInput(1));
  m_ForwardProjectionFilter->SetInput(1, this->GetInput(0));

  // Set memory management parameters for forward
  // and back projection filters
  m_ForwardProjectionFilter->ReleaseDataFlagOn();
  m_BackProjectionFilter->ReleaseDataFlagOn();

  // Have the last filter calculate its output information
  m_BackProjectionFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_BackProjectionFilter->GetOutput() );
}

template< typename TOutputImage >
void SIRTConjugateGradientOperator<TOutputImage>::GenerateData()
{
  // Execute Pipeline
  m_BackProjectionFilter->Update();

  // Get the output
  this->GraftOutput( m_BackProjectionFilter->GetOutput() );
}

}// end namespace


#endif
