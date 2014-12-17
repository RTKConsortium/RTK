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

#ifndef __rtkImagXRawToAttenuationImageFilter_txx
#define __rtkImagXRawToAttenuationImageFilter_txx

#include <itkImageFileWriter.h>

namespace rtk
{

template<class TOutputImage, unsigned char bitShift>
void
ImagXRawToAttenuationImageFilter<TOutputImage, bitShift>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer inputPtr =
    const_cast< InputImageType * >(this->GetInput());
  if ( !inputPtr )
    return;

  m_CropFilter->SetInput(inputPtr); //SR: this is most likely useless
  m_LookupTableFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion() );
  m_LookupTableFilter->GetOutput()->PropagateRequestedRegion();
}

template<class TOutputImage, unsigned char bitShift>
ImagXRawToAttenuationImageFilter<TOutputImage, bitShift>
::ImagXRawToAttenuationImageFilter()
{
  m_CropFilter = CropFilterType::New();
  m_BinningFilter = BinningFilterType::New();
  m_ScatterFilter = ScatterFilterType::New();
  m_I0estimationFilter = I0FilterType::New();
  m_LookupTableFilter = LookupTableFilterType::New();

  //Permanent internal connections
  m_BinningFilter->SetInput( m_CropFilter->GetOutput() )
  m_ScatterFilter->SetInput( m_BinningFilter->GetOutput() );
  m_I0estimationFilter->SetInput( m_ScatterFilter->GetOutput() );
  m_LookupTableFilter->SetInput( m_I0estimationFilter->GetOutput() );

  //Default filter parameters
  typename CropFilterType::SizeType border = m_CropFilter->GetLowerBoundaryCropSize();
  border[0] = 4;
  border[1] = 4;
  m_CropFilter->SetBoundaryCropSize(border);
}

template<class TOutputImage, unsigned char bitShift>
void
ImagXRawToAttenuationImageFilter<TOutputImage, bitShift>
::GenerateOutputInformation()
{
  m_CropFilter->SetInput(this->GetInput() );
  m_LookupTableFilter->UpdateOutputInformation();
  this->GetOutput()->SetOrigin( m_LookupTableFilter->GetOutput()->GetOrigin() );
  this->GetOutput()->SetSpacing( m_LookupTableFilter->GetOutput()->GetSpacing() );
  this->GetOutput()->SetDirection( m_LookupTableFilter->GetOutput()->GetDirection() );
  this->GetOutput()->SetLargestPossibleRegion( m_LookupTableFilter->GetOutput()->GetLargestPossibleRegion() );
}

template<class TOutputImage, unsigned char bitShift>
void
ImagXRawToAttenuationImageFilter<TOutputImage, bitShift>
::GenerateData()
{
  m_CropFilter->SetInput(this->GetInput() );
  m_LookupTableFilter->Update();
  this->GraftOutput( m_LookupTableFilter->GetOutput() );
}

} // end namespace rtk
#endif
