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
#ifndef rtkFourDToProjectionStackImageFilter_hxx
#define rtkFourDToProjectionStackImageFilter_hxx

#include "rtkFourDToProjectionStackImageFilter.h"
#include "rtkGeneralPurposeFunctions.h"

namespace rtk
{

template< typename ProjectionStackType, typename VolumeSeriesType>
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>::FourDToProjectionStackImageFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Create the filters that can be created (all but the forward projection filter)
  m_PasteFilter = PasteFilterType::New();
  m_InterpolationFilter = InterpolatorFilterType::New();
  m_ConstantVolumeSource = ConstantVolumeSourceType::New();
  m_ConstantProjectionStackSource = ConstantProjectionStackSourceType::New();

  // Set parameters
  m_PasteFilter->SetInPlace(true);

  // Set memory management flags
  m_InterpolationFilter->ReleaseDataFlagOn();
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::SetInputProjectionStack(const ProjectionStackType* Projection)
{
  this->SetNthInput(0, const_cast<ProjectionStackType*>(Projection));
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
  this->SetNthInput(1, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template< typename ProjectionStackType, typename VolumeSeriesType>
typename ProjectionStackType::Pointer
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::GetInputProjectionStack()
{
  return static_cast< ProjectionStackType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template< typename ProjectionStackType, typename VolumeSeriesType>
typename VolumeSeriesType::ConstPointer
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::GetInputVolumeSeries()
{
  return static_cast< const VolumeSeriesType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg)
{
  m_ForwardProjectionFilter = _arg;
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::SetWeights(const itk::Array2D<float> _arg)
{
  m_Weights = _arg;
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::SetGeometry(GeometryType::Pointer _arg)
{
  m_Geometry=_arg;
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDToProjectionStackImageFilter< VolumeSeriesType, ProjectionStackType>
::SetSignal(const std::vector<double> signal)
{
  this->m_Signal = signal;
  this->Modified();
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::InitializeConstantVolumeSource()
{
  // Set the volume source
  int VolumeDimension = VolumeType::ImageDimension;

  typename VolumeType::SizeType constantVolumeSourceSize;
  constantVolumeSourceSize.Fill(0);
  for(int i=0; i < VolumeDimension; i++)
      constantVolumeSourceSize[i] = GetInputVolumeSeries()->GetLargestPossibleRegion().GetSize()[i];

  typename VolumeType::SpacingType constantVolumeSourceSpacing;
  constantVolumeSourceSpacing.Fill(0);
  for(int i=0; i < VolumeDimension; i++)
      constantVolumeSourceSpacing[i] = GetInputVolumeSeries()->GetSpacing()[i];

  typename VolumeType::PointType constantVolumeSourceOrigin;
  constantVolumeSourceOrigin.Fill(0);
  for(int i=0; i < VolumeDimension; i++)
      constantVolumeSourceOrigin[i] = GetInputVolumeSeries()->GetOrigin()[i];

  typename VolumeType::DirectionType constantVolumeSourceDirection;
  constantVolumeSourceDirection.SetIdentity();

  m_ConstantVolumeSource->SetOrigin( constantVolumeSourceOrigin );
  m_ConstantVolumeSource->SetSpacing( constantVolumeSourceSpacing );
  m_ConstantVolumeSource->SetDirection( constantVolumeSourceDirection );
  m_ConstantVolumeSource->SetSize( constantVolumeSourceSize );
  m_ConstantVolumeSource->SetConstant( 0. );
  m_ConstantVolumeSource->Update();
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::GenerateOutputInformation()
{
  this->InitializeConstantVolumeSource();

  int ProjectionStackDimension = ProjectionStackType::ImageDimension;
  m_PasteRegion = this->GetInputProjectionStack()->GetLargestPossibleRegion();
  m_PasteRegion.SetSize(ProjectionStackDimension - 1, 1);

  // Set the projection stack source
  m_ConstantProjectionStackSource->SetInformationFromImage(this->GetInputProjectionStack());
  m_ConstantProjectionStackSource->SetSize(m_PasteRegion.GetSize());
  m_ConstantProjectionStackSource->SetConstant( 0. );

  // Connect the filters
  m_InterpolationFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());
  m_InterpolationFilter->SetInputVolume(m_ConstantVolumeSource->GetOutput());
  m_PasteFilter->SetDestinationImage(this->GetInputProjectionStack());

  // Connections with the Forward projection filter can only be set at runtime
  m_ForwardProjectionFilter->SetInput(0, m_ConstantProjectionStackSource->GetOutput());
  m_ForwardProjectionFilter->SetInput(1, m_InterpolationFilter->GetOutput());
  m_PasteFilter->SetSourceImage(m_ForwardProjectionFilter->GetOutput());

  // Set runtime parameters
  m_InterpolationFilter->SetWeights(m_Weights);
  m_InterpolationFilter->SetProjectionNumber(m_PasteRegion.GetIndex(ProjectionStackDimension - 1));
  m_ForwardProjectionFilter->SetGeometry(m_Geometry);
  m_PasteFilter->SetSourceRegion(m_PasteRegion);
  m_PasteFilter->SetDestinationIndex(m_PasteRegion.GetIndex());

  // Have the last filter calculate its output information
  m_PasteFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation(m_PasteFilter->GetOutput());
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::GenerateInputRequestedRegion()
{
  // Input 0 is the stack of projections we update
  typename ProjectionStackType::Pointer  inputPtr0 = const_cast< ProjectionStackType * >( this->GetInput(0) );
  if ( !inputPtr0 )
    {
    return;
    }
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the volume series
  typename VolumeSeriesType::Pointer inputPtr1 = static_cast< VolumeSeriesType * >
            ( this->itk::ProcessObject::GetInput(1) );
  inputPtr1->SetRequestedRegionToLargestPossibleRegion();
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::GenerateData()
{
  int ProjectionStackDimension = ProjectionStackType::ImageDimension;

  int NumberProjs = this->GetInputProjectionStack()->GetRequestedRegion().GetSize(ProjectionStackDimension-1);
  int FirstProj = this->GetInputProjectionStack()->GetRequestedRegion().GetIndex(ProjectionStackDimension-1);

  bool firstProjectionProcessed = false;

  // Process the projections in order
  for (int proj = FirstProj; proj < FirstProj+NumberProjs; proj++)
    {
    // After the first update, we need to use the output as input.
    if(firstProjectionProcessed)
      {
      typename ProjectionStackType::Pointer pimg = this->m_PasteFilter->GetOutput();
      pimg->DisconnectPipeline();
      this->m_PasteFilter->SetDestinationImage( pimg );
      }

    // Update the paste region
    this->m_PasteRegion.SetIndex(ProjectionStackDimension-1, proj);

    // Set the projection stack source
    this->m_ConstantProjectionStackSource->SetIndex(this->m_PasteRegion.GetIndex());

    // Set the Paste Filter. Since its output has been disconnected
    // we need to set its RequestedRegion manually (it will never
    // be updated by a downstream filter)
    m_PasteFilter->SetSourceRegion(m_PasteRegion);
    m_PasteFilter->SetDestinationIndex(m_PasteRegion.GetIndex());
    m_PasteFilter->GetOutput()->SetRequestedRegion(m_PasteFilter->GetDestinationImage()->GetLargestPossibleRegion());

    // Set the Interpolation filter
    m_InterpolationFilter->SetProjectionNumber(proj);

    // Update the last filter
    m_PasteFilter->Update();

    // Update condition
    firstProjectionProcessed = true;
    }

  // Graft its output
  this->GraftOutput( m_PasteFilter->GetOutput() );
}

}// end namespace


#endif
