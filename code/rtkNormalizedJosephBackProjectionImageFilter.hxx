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

#ifndef rtkNormalizedJosephBackProjectionImageFilter_hxx
#define rtkNormalizedJosephBackProjectionImageFilter_hxx

namespace rtk
{

template <class TInputImage, class TOutputImage>
NormalizedJosephBackProjectionImageFilter<TInputImage,TOutputImage>
::NormalizedJosephBackProjectionImageFilter()
{
  // Create the filter
  m_AddFilter = AddFilterType::New();
  m_DivideFilter = DivideFilterType::New();
  m_ConstantProjectionSource = ConstantProjectionSourceType::New();
  m_ConstantVolumeSource = ConstantVolumeSourceType::New();
  m_JosephBackProjector = JosephBackProjectionFilterType::New();
  m_JosephBackProjectorOfConstantProjection = JosephBackProjectionFilterType::New();

  // Permanent internal connections
  m_JosephBackProjector->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_JosephBackProjectorOfConstantProjection->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_JosephBackProjectorOfConstantProjection->SetInput(1, m_ConstantProjectionSource->GetOutput());
  m_DivideFilter->SetInput(0, m_JosephBackProjector->GetOutput());
  m_DivideFilter->SetInput(1, m_JosephBackProjectorOfConstantProjection->GetOutput());
  m_AddFilter->SetInput(1, m_DivideFilter->GetOutput());

}

template <class TInputImage, class TOutputImage>
void
NormalizedJosephBackProjectionImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer inputPtr =
    const_cast< TInputImage * >( this->GetInput() );

  if ( !inputPtr )
    return;

  m_AddFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion() );
  m_AddFilter->GetOutput()->PropagateRequestedRegion();

}

template <class TInputImage, class TOutputImage>
void
NormalizedJosephBackProjectionImageFilter<TInputImage,TOutputImage>
::GenerateOutputInformation()
{
  // Set runtime connections
  m_JosephBackProjector->SetInput ( 1, this->GetInput(1) );
  m_AddFilter->SetInput(0, this->GetInput(0));

  // Set geometry
  GeometryType *geometry = dynamic_cast<GeometryType*>(this->GetGeometry().GetPointer());
  m_JosephBackProjector->SetGeometry(geometry);
  m_JosephBackProjectorOfConstantProjection->SetGeometry(geometry);

  // Set constant image sources
  m_ConstantVolumeSource->SetInformationFromImage(const_cast<TInputImage *>(this->GetInput(0)));
  m_ConstantVolumeSource->SetConstant(0);
  m_ConstantVolumeSource->Update();

  m_ConstantProjectionSource->SetInformationFromImage(const_cast<TInputImage *>(this->GetInput(1)));
  m_ConstantProjectionSource->SetConstant(1);
  m_ConstantProjectionSource->UpdateOutputInformation();

  m_AddFilter->UpdateOutputInformation();

  // Update output information
  this->GetOutput()->SetOrigin( m_AddFilter->GetOutput()->GetOrigin() );
  this->GetOutput()->SetSpacing( m_AddFilter->GetOutput()->GetSpacing() );
  this->GetOutput()->SetDirection( m_AddFilter->GetOutput()->GetDirection() );
  this->GetOutput()->SetLargestPossibleRegion( m_AddFilter->GetOutput()->GetLargestPossibleRegion() );

}

template <class TInputImage, class TOutputImage>
void
NormalizedJosephBackProjectionImageFilter<TInputImage,TOutputImage>
::GenerateData()
{
    m_AddFilter->Update();
    this->GraftOutput( m_AddFilter->GetOutput());
}


} // end namespace rtk

#endif
