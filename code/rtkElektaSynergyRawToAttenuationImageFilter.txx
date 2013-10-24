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

#ifndef __rtkElektaSynergyRawToAttenuationImageFilter_txx
#define __rtkElektaSynergyRawToAttenuationImageFilter_txx

#include <itkImageFileWriter.h>

namespace rtk
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
  m_LogLookupTableFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion() );
  m_LogLookupTableFilter->GetOutput()->PropagateRequestedRegion();
}

template <class TInputImage, class TOutputImage>
ElektaSynergyRawToAttenuationImageFilter<TInputImage, TOutputImage>
::ElektaSynergyRawToAttenuationImageFilter()
{
  m_CropFilter = CropFilterType::New();
  m_RawLookupTableFilter = RawLookupTableFilterType::New();
  m_ScatterFilter = ScatterFilterType::New();
  m_LogLookupTableFilter = LogLookupTableFilterType::New();

  //Permanent internal connections
  m_RawLookupTableFilter->SetInput( m_CropFilter->GetOutput() );
  m_ScatterFilter->SetInput( m_RawLookupTableFilter->GetOutput() );
  m_LogLookupTableFilter->SetInput( m_ScatterFilter->GetOutput() );

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
  m_LogLookupTableFilter->UpdateOutputInformation();
  this->GetOutput()->SetOrigin( m_LogLookupTableFilter->GetOutput()->GetOrigin() );
  this->GetOutput()->SetSpacing( m_LogLookupTableFilter->GetOutput()->GetSpacing() );
  this->GetOutput()->SetDirection( m_LogLookupTableFilter->GetOutput()->GetDirection() );
  this->GetOutput()->SetLargestPossibleRegion( m_LogLookupTableFilter->GetOutput()->GetLargestPossibleRegion() );
}

template<class TInputImage, class TOutputImage>
void
ElektaSynergyRawToAttenuationImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  m_CropFilter->SetInput(this->GetInput() );
  m_LogLookupTableFilter->Update();
  this->GraftOutput( m_LogLookupTableFilter->GetOutput() );
}

} // end namespace rtk
#endif
