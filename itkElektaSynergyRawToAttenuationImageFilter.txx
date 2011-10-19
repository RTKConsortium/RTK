#ifndef __itkElektaSynergyRawToAttenuationImageFilter_txx
#define __itkElektaSynergyRawToAttenuationImageFilter_txx

#include <itkImageFileWriter.h>

namespace itk
{

template<class TInputImage, class TOutputImage>
void
ElektaSynergyRawToAttenuationImageFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer inputPtr =
    const_cast< TInputImage * >( this->GetInput() );
  if ( !inputPtr )
    return;

  m_CropFilter->SetInput(inputPtr); //SR: this is most likely useless
  m_LutFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion() );
  m_LutFilter->GetOutput()->PropagateRequestedRegion();
}

template <class TInputImage, class TOutputImage>
ElektaSynergyRawToAttenuationImageFilter<TInputImage, TOutputImage>
::ElektaSynergyRawToAttenuationImageFilter()
{
  m_CropFilter = CropFilterType::New();
  m_ScatterFilter = ScatterFilterType::New();
  m_LutFilter = LutFilterType::New();

  //Permanent internal connections
  m_ScatterFilter->SetInput( m_CropFilter->GetOutput() );
  m_LutFilter->SetInput( m_ScatterFilter->GetOutput() );

  //Default filter parameters
  typename CropFilterType::SizeType border = m_CropFilter->GetLowerBoundaryCropSize();
  border[0] = 4;
  border[1] = 4;
  m_CropFilter->SetBoundaryCropSize(border);
}

template<class TInputImage, class TOutputImage>
void
ElektaSynergyRawToAttenuationImageFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{
  m_CropFilter->SetInput(this->GetInput() );
  m_LutFilter->UpdateOutputInformation();
  this->GetOutput()->SetOrigin( m_LutFilter->GetOutput()->GetOrigin() );
  this->GetOutput()->SetSpacing( m_LutFilter->GetOutput()->GetSpacing() );
  this->GetOutput()->SetDirection( m_LutFilter->GetOutput()->GetDirection() );
  this->GetOutput()->SetLargestPossibleRegion( m_LutFilter->GetOutput()->GetLargestPossibleRegion() );
}

template<class TInputImage, class TOutputImage>
void
ElektaSynergyRawToAttenuationImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  m_CropFilter->SetInput(this->GetInput() );
  m_LutFilter->Update();
  this->GraftOutput( m_LutFilter->GetOutput() );
}

} // end namespace itk
#endif
