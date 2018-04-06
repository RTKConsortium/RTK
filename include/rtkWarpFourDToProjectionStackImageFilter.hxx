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
#ifndef rtkWarpFourDToProjectionStackImageFilter_hxx
#define rtkWarpFourDToProjectionStackImageFilter_hxx

#include "rtkWarpFourDToProjectionStackImageFilter.h"

namespace rtk
{

template< typename VolumeSeriesType, typename ProjectionStackType>
WarpFourDToProjectionStackImageFilter< VolumeSeriesType, ProjectionStackType>::WarpFourDToProjectionStackImageFilter():
    m_UseCudaCyclicDeformation(false)
{
  this->SetNumberOfRequiredInputs(3);

#ifdef RTK_USE_CUDA
  this->m_ForwardProjectionFilter = rtk::CudaWarpForwardProjectionImageFilter::New();
#else
  this->m_ForwardProjectionFilter = rtk::JosephForwardProjectionImageFilter<ProjectionStackType, ProjectionStackType>::New();
  itkWarningMacro("The warp Forward project image filter exists only in CUDA. Ignoring the displacement vector field and using CPU Joseph forward projection")
#endif
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
WarpFourDToProjectionStackImageFilter< VolumeSeriesType, ProjectionStackType>::SetDisplacementField(const DVFSequenceImageType* DisplacementField)
{
  this->SetNthInput(2, const_cast<DVFSequenceImageType*>(DisplacementField));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename WarpFourDToProjectionStackImageFilter< VolumeSeriesType, ProjectionStackType>::DVFSequenceImageType::ConstPointer
WarpFourDToProjectionStackImageFilter< VolumeSeriesType, ProjectionStackType>::GetDisplacementField()
{
  return static_cast< const DVFSequenceImageType * >
          ( this->itk::ProcessObject::GetInput(2) );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
WarpFourDToProjectionStackImageFilter< VolumeSeriesType, ProjectionStackType>
::SetSignal(const std::vector<double> signal)
{
  this->m_Signal = signal;
  this->Modified();
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
WarpFourDToProjectionStackImageFilter< VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  m_DVFInterpolatorFilter = DVFInterpolatorType::New();
#ifdef RTK_USE_CUDA
  if (m_UseCudaCyclicDeformation)
    m_DVFInterpolatorFilter = rtk::CudaCyclicDeformationImageFilter::New();
#endif
  m_DVFInterpolatorFilter->SetSignalVector(m_Signal);
  m_DVFInterpolatorFilter->SetInput(this->GetDisplacementField());
  m_DVFInterpolatorFilter->SetFrame(0);

#ifdef RTK_USE_CUDA
  dynamic_cast<rtk::CudaWarpForwardProjectionImageFilter* >
      (this->m_ForwardProjectionFilter.GetPointer())->SetDisplacementField(m_DVFInterpolatorFilter->GetOutput());
#endif

  Superclass::GenerateOutputInformation();
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
WarpFourDToProjectionStackImageFilter< VolumeSeriesType, ProjectionStackType>
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

  // Input 2 is the sequence of DVFs
  typename DVFSequenceImageType::Pointer inputPtr2 = static_cast< DVFSequenceImageType * >
            ( this->itk::ProcessObject::GetInput(2) );
  inputPtr2->SetRequestedRegionToLargestPossibleRegion();
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
WarpFourDToProjectionStackImageFilter< VolumeSeriesType, ProjectionStackType>
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
    this->m_PasteFilter->SetSourceRegion(this->m_PasteRegion);
    this->m_PasteFilter->SetDestinationIndex(this->m_PasteRegion.GetIndex());
    this->m_PasteFilter->GetOutput()->SetRequestedRegion(this->m_PasteFilter->GetDestinationImage()->GetLargestPossibleRegion());

    // Set the Interpolation filter
    this->m_InterpolationFilter->SetProjectionNumber(proj);

    // Set the DVF interpolator
    m_DVFInterpolatorFilter->SetFrame(proj);

    // Update the last filter
    this->m_PasteFilter->Update();

    // Update condition
    firstProjectionProcessed = true;
    }

  // Graft its output
  this->GraftOutput( this->m_PasteFilter->GetOutput() );

  // Release the data in internal filters
  this->m_DVFInterpolatorFilter->GetOutput()->ReleaseData();
}

}// end namespace


#endif
