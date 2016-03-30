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
#ifndef __rtkMotionCompensatedFourDReconstructionConjugateGradientOperator_hxx
#define __rtkMotionCompensatedFourDReconstructionConjugateGradientOperator_hxx

#include "rtkMotionCompensatedFourDReconstructionConjugateGradientOperator.h"

namespace rtk
{

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::MotionCompensatedFourDReconstructionConjugateGradientOperator()
{
  this->SetNumberOfRequiredInputs(3);
  m_MVFInterpolatorFilter = MVFInterpolatorType::New();
  m_InverseMVFInterpolatorFilter = MVFInterpolatorType::New();
}


template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::SetDisplacementField(const TMVFImageSequence* DisplacementField)
{
  this->SetNthInput(2, const_cast<TMVFImageSequence*>(DisplacementField));
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::SetInverseDisplacementField(const TMVFImageSequence* InverseDisplacementField)
{
  this->SetNthInput(3, const_cast<TMVFImageSequence*>(InverseDisplacementField));
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
typename TMVFImageSequence::ConstPointer
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GetDisplacementField()
{
  return static_cast< const TMVFImageSequence * >
          ( this->itk::ProcessObject::GetInput(2) );
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
typename TMVFImageSequence::ConstPointer
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GetInverseDisplacementField()
{
  return static_cast< const TMVFImageSequence * >
          ( this->itk::ProcessObject::GetInput(3) );
}

#ifdef RTK_USE_CUDA
template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
CudaWarpForwardProjectionImageFilter *
MotionCompensatedFourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GetForwardProjectionFilter()
{
  return(m_ForwardProjectionFilter.GetPointer());
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
CudaWarpBackProjectionImageFilter *
MotionCompensatedFourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GetBackProjectionFilter()
{
  return(m_BackProjectionFilter.GetPointer());
}
#endif

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
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
  m_InverseMVFInterpolatorFilter->SetSignalVector(m_Signal);
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GenerateOutputInformation()
{
  m_MVFInterpolatorFilter->SetInput(this->GetDisplacementField());
  m_MVFInterpolatorFilter->SetFrame(0);

  m_InverseMVFInterpolatorFilter->SetInput(this->GetInverseDisplacementField());
  m_InverseMVFInterpolatorFilter->SetFrame(0);

#ifdef RTK_USE_CUDA
  this->m_ForwardProjectionFilter = rtk::CudaWarpForwardProjectionImageFilter::New();
  GetForwardProjectionFilter()->SetDisplacementField(m_InverseMVFInterpolatorFilter->GetOutput());

  this->m_BackProjectionFilter = rtk::CudaWarpBackProjectionImageFilter::New();
  GetBackProjectionFilter()->SetDisplacementField(m_MVFInterpolatorFilter->GetOutput());
#else
  this->m_BackProjectionFilter = rtk::BackProjectionImageFilter<VolumeType, VolumeType>::New();
  this->m_ForwardProjectionFilter = rtk::JosephForwardProjectionImageFilter<ProjectionStackType, ProjectionStackType>::New();
  itkWarningMacro("The warp forward and back project image filters exist only in CUDA. Ignoring the displacement vector field and using CPU Joseph forward projection and CPU voxel-based back projection")
#endif

  Superclass::GenerateOutputInformation();
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GenerateInputRequestedRegion()
{
  // Let the internal filters compute the input requested region
  this->m_SplatFilter->PropagateRequestedRegion(this->m_SplatFilter->GetOutput());

  // The projection stack need not be loaded in memory, is it only used to configure the
  // constantProjectionStackSource with the correct information
  // Leave its requested region unchanged (set by the other filters that need it)
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GenerateData()
{
  int Dimension = ProjectionStackType::ImageDimension;

  // Prepare the index for the constant projection stack source
  typename ProjectionStackType::IndexType ConstantProjectionStackSourceIndex
      = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetIndex();

  int NumberProjs = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(2);
  typename VolumeSeriesType::Pointer pimg;
  for(int proj=0; proj<NumberProjs; proj++)
    {
    // After the first update, we need to use the output as input.
    if(proj>0)
      {
      pimg = this->m_SplatFilter->GetOutput();
      pimg->DisconnectPipeline();
      this->m_SplatFilter->SetInputVolumeSeries( pimg );
      }

    // Set the projection stack source
    ConstantProjectionStackSourceIndex[Dimension - 1] = proj;
    this->m_ConstantProjectionStackSource->SetIndex( ConstantProjectionStackSourceIndex );

    // Set the Interpolation filter
    this->m_InterpolationFilter->SetProjectionNumber(proj);
    this->m_SplatFilter->SetProjectionNumber(proj);

    // Set the MVF interpolator
    m_MVFInterpolatorFilter->SetFrame(proj);
    m_InverseMVFInterpolatorFilter->SetFrame(proj);

    // Update the last filter
    this->m_SplatFilter->Update();
    }

  // Graft its output
  this->GraftOutput( this->m_SplatFilter->GetOutput() );

  // Release the data in internal filters
  pimg->ReleaseData();
  this->m_ConstantVolumeSource1->GetOutput()->ReleaseData();
  this->m_ConstantVolumeSource2->GetOutput()->ReleaseData();
  this->m_ConstantVolumeSeriesSource->GetOutput()->ReleaseData();
  this->m_ConstantProjectionStackSource->GetOutput()->ReleaseData();
  this->m_DisplacedDetectorFilter->GetOutput()->ReleaseData();
  this->m_InterpolationFilter->GetOutput()->ReleaseData();
  this->GetBackProjectionFilter()->GetOutput()->ReleaseData();
  this->GetForwardProjectionFilter()->GetOutput()->ReleaseData();
  m_MVFInterpolatorFilter->GetOutput()->ReleaseData();
  m_InverseMVFInterpolatorFilter->GetOutput()->ReleaseData();

  // Send the input back onto the CPU
  this->GetInputVolumeSeries()->GetBufferPointer();
}

}// end namespace


#endif
