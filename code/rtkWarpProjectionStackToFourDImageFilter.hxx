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
#ifndef __rtkWarpProjectionStackToFourDImageFilter_hxx
#define __rtkWarpProjectionStackToFourDImageFilter_hxx

#include "rtkWarpProjectionStackToFourDImageFilter.h"

#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

namespace rtk
{

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
WarpProjectionStackToFourDImageFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>::WarpProjectionStackToFourDImageFilter()
{
  this->SetNumberOfRequiredInputs(3);

  m_MVFInterpolatorFilter = MVFInterpolatorType::New();
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
WarpProjectionStackToFourDImageFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>::SetDisplacementField(const TMVFImageSequence* DisplacementField)
{
  this->SetNthInput(2, const_cast<TMVFImageSequence*>(DisplacementField));
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
typename TMVFImageSequence::ConstPointer
WarpProjectionStackToFourDImageFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>::GetDisplacementField()
{
  return static_cast< const TMVFImageSequence * >
          ( this->itk::ProcessObject::GetInput(2) );
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
CudaWarpBackProjectionImageFilter *
WarpProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>::GetBackProjectionFilter()
{
  return(m_BackProjectionFilter.GetPointer());
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
WarpProjectionStackToFourDImageFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::SetSignalFilename(const std::string _arg)
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

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
WarpProjectionStackToFourDImageFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GenerateOutputInformation()
{
  m_MVFInterpolatorFilter->SetInput(this->GetDisplacementField());
  m_MVFInterpolatorFilter->SetFrame(0);

#ifdef RTK_USE_CUDA
  this->m_BackProjectionFilter = rtk::CudaWarpBackProjectionImageFilter::New();
  GetBackProjectionFilter()->SetDisplacementField(m_MVFInterpolatorFilter->GetOutput());
#else
  m_BackProjectionFilter = rtk::BackProjectionImageFilter<VolumeType, VolumeType>::New();
  itkWarningMacro("The warp back project image filter exists only in CUDA. Ignoring the displacement vector field and using CPU voxel-based back projection")
#endif

  Superclass::GenerateOutputInformation();
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
WarpProjectionStackToFourDImageFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GenerateData()
{
  int Dimension = ProjectionStackType::ImageDimension;

  // Set the Extract filter
  typename ProjectionStackType::RegionType extractRegion;
  extractRegion = this->GetInputProjectionStack()->GetLargestPossibleRegion();
  extractRegion.SetSize(Dimension-1, 1);

  // Declare the pointer to a VolumeSeries that will be used in the pipeline
  typename VolumeSeriesType::Pointer pimg;

  int NumberProjs = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(Dimension-1);
  int FirstProj = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetIndex(Dimension-1);
  for(this->m_ProjectionNumber=FirstProj; this->m_ProjectionNumber<FirstProj+NumberProjs; this->m_ProjectionNumber++)
    {
    // After the first update, we need to use the output as input.
    if(this->m_ProjectionNumber>FirstProj)
      {
      pimg = this->m_SplatFilter->GetOutput();
      pimg->DisconnectPipeline();
      this->m_SplatFilter->SetInputVolumeSeries( pimg );
      }

    // Set the Extract Filter
    extractRegion.SetIndex(Dimension-1, this->m_ProjectionNumber);
    this->m_ExtractFilter->SetExtractionRegion(extractRegion);

    // Set the MVF interpolator
    m_MVFInterpolatorFilter->SetFrame(this->m_ProjectionNumber);

    // Set the splat filter
    this->m_SplatFilter->SetProjectionNumber(this->m_ProjectionNumber);

    // Update the last filter
    this->m_SplatFilter->Update();
    }

  // Graft its output
  this->GraftOutput( this->m_SplatFilter->GetOutput() );

  // Release the data in internal filters
  if(pimg.IsNotNull())
    pimg->ReleaseData();
  this->m_DisplacedDetectorFilter->GetOutput()->ReleaseData();
  this->m_BackProjectionFilter->GetOutput()->ReleaseData();
  this->m_ExtractFilter->GetOutput()->ReleaseData();
  this->m_ConstantVolumeSource->GetOutput()->ReleaseData();
  this->m_MVFInterpolatorFilter->GetOutput()->ReleaseData();
}

}// end namespace


#endif
