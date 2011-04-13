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

  m_LutFilter->SetInput(inputPtr); //SR: this is most likely useless
  m_CropFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion() );
  m_CropFilter->GetOutput()->PropagateRequestedRegion();
}

template <class TInputImage, class TOutputImage>
ElektaSynergyRawToAttenuationImageFilter<TInputImage, TOutputImage>
::ElektaSynergyRawToAttenuationImageFilter()
{
  m_LutFilter = LutFilterType::New();
  m_CropFilter = CropFilterType::New();

  //Permanent internal connections
  m_CropFilter->SetInput(m_LutFilter->GetOutput() );

  //Default filter parameters
  typename CropFilterType::SizeType border = m_CropFilter->GetLowerBoundaryCropSize();
  border[0] = 4;
  m_CropFilter->SetBoundaryCropSize(border);
}

template<class TInputImage, class TOutputImage>
void
ElektaSynergyRawToAttenuationImageFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{
  m_LutFilter->SetInput(this->GetInput() );
  m_CropFilter->UpdateOutputInformation();
  this->GetOutput()->SetOrigin( m_CropFilter->GetOutput()->GetOrigin() );
  this->GetOutput()->SetSpacing( m_CropFilter->GetOutput()->GetSpacing() );
  this->GetOutput()->SetDirection( m_CropFilter->GetOutput()->GetDirection() );
  this->GetOutput()->SetLargestPossibleRegion( m_CropFilter->GetOutput()->GetLargestPossibleRegion() );
}

template<class TInputImage, class TOutputImage>
void
ElektaSynergyRawToAttenuationImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  this->AllocateOutputs();
  m_LutFilter->SetInput(this->GetInput() );
  m_CropFilter->Update();
  this->GraftOutput( m_CropFilter->GetOutput() );
}

} // end namespace itk
#endif
