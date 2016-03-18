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

#ifndef __rtkWarpForwardProjectSequenceImageFilter_hxx
#define __rtkWarpForwardProjectSequenceImageFilter_hxx

#include "rtkWarpForwardProjectSequenceImageFilter.h"

#include <itkImageFileWriter.h>
#include <iostream>
#include <sstream>
#include <string>

namespace rtk
{

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
WarpForwardProjectSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::WarpForwardProjectSequenceImageFilter()
{
  this->SetNumberOfRequiredInputs(1);

  // Create the filters
  m_MVFInterpolatorFilter = MVFInterpolatorType::New();
  m_PasteFilter = PasteFilterType::New();
  m_SingleProjectionSource = ConstantImageSourceType::New();
  m_ProjectionStackSource = ConstantImageSourceType::New();
  m_VolumeSource = ConstantImageSourceType::New();

#ifdef RTK_USE_CUDA
  m_InterpolatorFilter = CudaInterpolateFilterType::New();
  m_ForwardProjectFilter = CudaWarpForwardProjectFilterType::New();
#else
  itkWarningMacro(<< "The warp forward project image filter exists only in CUDA. Ignoring the displacement vector field and using Joseph forward projection");
  m_InterpolatorFilter = InterpolateFilterType::New();
  m_ForwardProjectFilter = JosephForwardProjectFilterType::New();
#endif

  // Set memory management parameters
  m_PasteFilter->SetInPlace(true);
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
void
WarpForwardProjectSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::SetSignalFilename (const std::string _arg)
{
  itkDebugMacro("setting SignalFilename to " << _arg);
  if ( this->m_SignalFilename != _arg )
    {
    this->m_SignalFilename = _arg;
    this->Modified();

    std::ifstream is( _arg.c_str() );
    if( !is.is_open() )
      {
      itkGenericExceptionMacro(<< "Could not open signal file " << m_SignalFilename);
      }

    double value;
    while( !is.eof() )
      {
      is >> value;
      m_Signal.push_back(value);
      }
    }
  m_MVFInterpolatorFilter->SetSignalVector(m_Signal);
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
void
WarpForwardProjectSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::SetInputProjectionStack(const ProjectionStackType* Projection)
{
  this->SetPrimaryInput(const_cast<ProjectionStackType*>(Projection));
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
void
WarpForwardProjectSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::SetInputVolumeSeries(const TImageSequence* VolumeSeries)
{
  this->SetInput("VolumeSeries", const_cast<TImageSequence*>(VolumeSeries));
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
void
WarpForwardProjectSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::SetDisplacementField(const TMVFImageSequence* MVFs)
{
  this->SetInput("DisplacementField", const_cast<TMVFImageSequence*>(MVFs));
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
typename TImage::Pointer
WarpForwardProjectSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::GetInputProjectionStack()
{
  return static_cast< ProjectionStackType * >
          ( this->itk::ProcessObject::GetInput("Primary") );
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
typename TImageSequence::Pointer
WarpForwardProjectSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::GetInputVolumeSeries()
{
  return static_cast< TImageSequence * >
          ( this->itk::ProcessObject::GetInput("VolumeSeries") );
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
typename TMVFImageSequence::Pointer
WarpForwardProjectSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::GetDisplacementField()
{
  return static_cast< TMVFImageSequence * >
          ( this->itk::ProcessObject::GetInput("DisplacementField") );
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
void
WarpForwardProjectSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::SetWeights(const itk::Array2D<float> _arg)
{
  m_InterpolatorFilter->SetWeights(_arg);
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
void
WarpForwardProjectSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::GenerateOutputInformation()
{
  int Dimension = TImage::ImageDimension;

  // Region for the single projection source and the paste filter
  m_SingleProjectionRegion = this->GetInputProjectionStack()->GetLargestPossibleRegion();
  m_SingleProjectionRegion.SetSize(Dimension - 1, 1);
  m_SingleProjectionRegion.SetIndex(Dimension - 1, 0);

  // Set the sources
  m_ProjectionStackSource->SetInformationFromImage(this->GetInputProjectionStack());

  m_SingleProjectionSource->SetInformationFromImage(this->GetInputProjectionStack());
  m_SingleProjectionSource->SetSize(m_SingleProjectionRegion.GetSize());
  m_SingleProjectionSource->SetIndex(m_SingleProjectionRegion.GetIndex());

  typename TImage::SizeType VolumeSize;
  VolumeSize.Fill(0);
  typename TImage::SpacingType VolumeSpacing;
  VolumeSpacing.Fill(0);
  typename TImage::PointType VolumeOrigin;
  VolumeOrigin.Fill(0);
  typename TImage::IndexType VolumeIndex;
  VolumeIndex.Fill(0);
  typename TImage::DirectionType VolumeDirection;
  VolumeDirection.Fill(0);
  for (int i=0; i<Dimension; i++)
    {
    VolumeSize[i] = this->GetInputVolumeSeries()->GetLargestPossibleRegion().GetSize()[i];
    VolumeIndex[i] = this->GetInputVolumeSeries()->GetLargestPossibleRegion().GetIndex()[i];
    VolumeSpacing[i] = this->GetInputVolumeSeries()->GetSpacing()[i];
    VolumeOrigin[i] = this->GetInputVolumeSeries()->GetOrigin()[i];
    for (int j=0; j<Dimension; j++)
      {
      VolumeDirection(i, j) = this->GetInputVolumeSeries()->GetDirection()(i, j);
      }
    }
  m_VolumeSource->SetSize(VolumeSize);
  m_VolumeSource->SetSpacing(VolumeSpacing);
  m_VolumeSource->SetOrigin(VolumeOrigin);
  m_VolumeSource->SetIndex(VolumeIndex);
  m_VolumeSource->SetDirection(VolumeDirection);

  // Connect the filters
  m_InterpolatorFilter->SetInputVolume(m_VolumeSource->GetOutput());
  m_InterpolatorFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());
  m_InterpolatorFilter->SetProjectionNumber(0);

  m_MVFInterpolatorFilter->SetInput(this->GetDisplacementField());
  m_MVFInterpolatorFilter->SetFrame(0);

  m_ForwardProjectFilter->SetGeometry(this->m_Geometry);
#ifdef RTK_USE_CUDA
  m_ForwardProjectFilter->SetInputVolume(m_InterpolatorFilter->GetOutput());
  m_ForwardProjectFilter->SetInputProjectionStack(m_SingleProjectionSource->GetOutput());
  m_ForwardProjectFilter->SetDisplacementField(m_MVFInterpolatorFilter->GetOutput());
#else
  m_ForwardProjectFilter->SetInput(0, m_SingleProjectionSource->GetOutput());
  m_ForwardProjectFilter->SetInput(1, m_InterpolatorFilter->GetOutput());
#endif

  m_PasteFilter->SetSourceImage(m_ForwardProjectFilter->GetOutput());
  m_PasteFilter->SetDestinationImage(m_ProjectionStackSource->GetOutput());

  m_ForwardProjectFilter->UpdateOutputInformation();

  m_PasteFilter->SetSourceRegion(m_ForwardProjectFilter->GetOutput()->GetLargestPossibleRegion());
  m_PasteFilter->SetDestinationIndex(m_SingleProjectionRegion.GetIndex());

  // Have the last filter calculate its output information
  m_PasteFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_PasteFilter->GetOutput() );
}


template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
void
WarpForwardProjectSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::GenerateInputRequestedRegion()
{
  //Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  //The input projection stack is only useful for its meta data. Yet setting its requested
  //region to empty may cause trouble if it is used as input for other filters at the same time
  //So we leave it as is
  this->GetInputVolumeSeries()->SetRequestedRegionToLargestPossibleRegion();
  this->GetDisplacementField()->SetRequestedRegionToLargestPossibleRegion();
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
void
WarpForwardProjectSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::GenerateData()
{
  int Dimension = TImage::ImageDimension;

  // Declare an image pointer to disconnect the output of paste
  typename TImage::Pointer pimg;

  for (unsigned int proj=0; proj<this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(Dimension-1); proj++)
    {
    if (proj > 0) // After the first proj, use the output of paste as input
      {
      pimg = m_PasteFilter->GetOutput();
      pimg->DisconnectPipeline();
      m_PasteFilter->SetDestinationImage(pimg);
      }

    m_SingleProjectionRegion.SetIndex(Dimension - 1, proj);
    m_MVFInterpolatorFilter->SetFrame(proj);
    m_InterpolatorFilter->SetProjectionNumber(proj);
    m_SingleProjectionSource->SetIndex(m_SingleProjectionRegion.GetIndex());
    m_PasteFilter->SetDestinationIndex(m_SingleProjectionRegion.GetIndex());
    m_PasteFilter->SetSourceRegion(m_SingleProjectionRegion);

    m_PasteFilter->Update();
    }
  this->GraftOutput( m_PasteFilter->GetOutput() );

  m_SingleProjectionSource->GetOutput()->ReleaseData();
  m_MVFInterpolatorFilter->GetOutput()->ReleaseData();
  m_InterpolatorFilter->GetOutput()->ReleaseData();
}


}// end namespace


#endif
