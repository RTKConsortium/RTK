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

template< typename VolumeSeriesType, typename ProjectionStackType>
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>
::MotionCompensatedFourDReconstructionConjugateGradientOperator()
{
  this->SetNumberOfRequiredInputs(3);

#ifdef RTK_USE_CUDA
  m_DVFInterpolatorFilter = rtk::CudaCyclicDeformationImageFilter::New();
  m_InverseDVFInterpolatorFilter = rtk::CudaCyclicDeformationImageFilter::New();
  this->m_ForwardProjectionFilter = rtk::CudaWarpForwardProjectionImageFilter::New();
  this->m_BackProjectionFilter = rtk::CudaWarpBackProjectionImageFilter::New();
#else
  m_DVFInterpolatorFilter = DVFInterpolatorType::New();
  m_InverseDVFInterpolatorFilter = DVFInterpolatorType::New();
  this->m_BackProjectionFilter = rtk::BackProjectionImageFilter<VolumeType, VolumeType>::New();
  this->m_ForwardProjectionFilter = rtk::JosephForwardProjectionImageFilter<ProjectionStackType, ProjectionStackType>::New();
  itkWarningMacro("The warp forward and back project image filters exist only in CUDA. Ignoring the displacement vector field and using CPU Joseph forward projection and CPU voxel-based back projection")
#endif

}


template< typename VolumeSeriesType, typename ProjectionStackType>
void
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>
::SetDisplacementField(const DVFSequenceImageType* DisplacementField)
{
  this->SetNthInput(2, const_cast<DVFSequenceImageType*>(DisplacementField));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>
::SetInverseDisplacementField(const DVFSequenceImageType* InverseDisplacementField)
{
  this->SetNthInput(3, const_cast<DVFSequenceImageType*>(InverseDisplacementField));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>::DVFSequenceImageType::ConstPointer
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>
::GetDisplacementField()
{
  return static_cast< const DVFSequenceImageType * >
          ( this->itk::ProcessObject::GetInput(2) );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>::DVFSequenceImageType::ConstPointer
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>
::GetInverseDisplacementField()
{
  return static_cast< const DVFSequenceImageType * >
          ( this->itk::ProcessObject::GetInput(3) );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>
::SetSignal(const std::vector<double> signal)
{
  this->m_Signal = signal;
  m_DVFInterpolatorFilter->SetSignalVector(signal);
  m_InverseDVFInterpolatorFilter->SetSignalVector(signal);
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  m_DVFInterpolatorFilter->SetInput(this->GetDisplacementField());
  m_DVFInterpolatorFilter->SetFrame(0);

  m_InverseDVFInterpolatorFilter->SetInput(this->GetInverseDisplacementField());
  m_InverseDVFInterpolatorFilter->SetFrame(0);

#ifdef RTK_USE_CUDA
  dynamic_cast< CudaWarpForwardProjectionImageFilter* >(this->m_ForwardProjectionFilter.GetPointer())->SetDisplacementField(m_InverseDVFInterpolatorFilter->GetOutput());
  dynamic_cast< CudaWarpBackProjectionImageFilter* >(this->m_BackProjectionFilter.GetPointer())->SetDisplacementField(m_DVFInterpolatorFilter->GetOutput());
#endif

  Superclass::GenerateOutputInformation();
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>
::GenerateData()
{
  int Dimension = ProjectionStackType::ImageDimension;

  // Prepare the index for the constant projection stack source
  typename ProjectionStackType::IndexType ConstantProjectionStackSourceIndex
      = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetIndex();

  int NumberProjs = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(Dimension-1);
  int FirstProj = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetIndex(Dimension-1);

  // Get an index permutation that sorts the signal values. Then process the projections
  // in that permutated order. This way, projections with identical phases will be
  // processed one after the other. This will save some of the DVF interpolation operations.
  std::vector<unsigned int> IndicesOfProjectionsSortedByPhase = GetSortingPermutation<double>(this->m_Signal);

  bool firstProjectionProcessed = false;
  typename VolumeSeriesType::Pointer pimg;
  int proj = 0;

  // Process the projections in permutated order
  for (unsigned int i = 0 ; i < this->m_Signal.size(); i++)
    {
    // Make sure the current projection is in the input projection stack's largest possible region
    proj = IndicesOfProjectionsSortedByPhase[i];
    if ((proj >= FirstProj) && (proj<FirstProj+NumberProjs))
      {
      // After the first update, we need to use the output as input.
      if(firstProjectionProcessed)
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

      // Set the DVF interpolator
      m_DVFInterpolatorFilter->SetFrame(proj);
      m_InverseDVFInterpolatorFilter->SetFrame(proj);

      // Update the last filter
      this->m_SplatFilter->Update();

      // Update condition
      firstProjectionProcessed = true;
      }
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
  this->m_BackProjectionFilter->GetOutput()->ReleaseData();
  this->m_ForwardProjectionFilter->GetOutput()->ReleaseData();
  m_DVFInterpolatorFilter->GetOutput()->ReleaseData();
  m_InverseDVFInterpolatorFilter->GetOutput()->ReleaseData();

  // Send the input back onto the CPU
  this->GetInputVolumeSeries()->GetBufferPointer();
}

}// end namespace


#endif
