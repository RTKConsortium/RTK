#ifndef __itkElektaSynergyRawToAttenuationImageFilter_txx
#define __itkElektaSynergyRawToAttenuationImageFilter_txx

#include <itkImageFileWriter.h>

namespace itk
{

template <class TInputImage, class TOutputImage>
ElektaSynergyRawToAttenuationImageFilter<TInputImage, TOutputImage>
::ElektaSynergyRawToAttenuationImageFilter()
{
  m_LutFilter = LutFilterType::New();
}

template<class TInputImage, class TOutputImage>
void
ElektaSynergyRawToAttenuationImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  this->AllocateOutputs();
  m_LutFilter->SetInput(this->GetInput());
  m_LutFilter->Update();
  this->GraftOutput( m_LutFilter->GetOutput() );
}

} // end namespace itk
#endif
