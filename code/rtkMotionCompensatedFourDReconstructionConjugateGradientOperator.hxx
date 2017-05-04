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
#ifndef rtkMotionCompensatedFourDReconstructionConjugateGradientOperator_hxx
#define rtkMotionCompensatedFourDReconstructionConjugateGradientOperator_hxx

#include "rtkMotionCompensatedFourDReconstructionConjugateGradientOperator.h"

namespace rtk
{

template< typename VolumeSeriesType, typename ProjectionStackType>
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>
::MotionCompensatedFourDReconstructionConjugateGradientOperator()
{
  this->SetNumberOfRequiredInputs(2);

  m_UseCudaCyclicDeformation = false;

#ifdef RTK_USE_CUDA
  this->m_ForwardProjectionFilter = rtk::CudaWarpForwardProjectionImageFilter::New();
  this->m_BackProjectionFilter = rtk::CudaWarpBackProjectionImageFilter::New();
#else
  this->m_BackProjectionFilter = rtk::BackProjectionImageFilter<VolumeType, VolumeType>::New();
  this->m_ForwardProjectionFilter = rtk::JosephForwardProjectionImageFilter<ProjectionStackType, ProjectionStackType>::New();
  itkWarningMacro("The warp forward and back project image filters exist only"
          << " in CUDA. Ignoring the displacement vector field and using CPU"
          << "Joseph forward projection and CPU voxel-based back projection")
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
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
MotionCompensatedFourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  m_DVFInterpolatorFilter = DVFInterpolatorType::New();
  m_InverseDVFInterpolatorFilter = DVFInterpolatorType::New();
#ifdef RTK_USE_CUDA
  if (m_UseCudaCyclicDeformation)
    {
    m_DVFInterpolatorFilter = rtk::CudaCyclicDeformationImageFilter::New();
    m_InverseDVFInterpolatorFilter = rtk::CudaCyclicDeformationImageFilter::New();
    }
#endif
  m_DVFInterpolatorFilter->SetSignalVector(this->m_Signal);
  m_DVFInterpolatorFilter->SetInput(this->GetDisplacementField());
  m_DVFInterpolatorFilter->SetFrame(0);

  m_InverseDVFInterpolatorFilter->SetSignalVector(this->m_Signal);
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

  // Prepare the index for the constant projection stack source and the extract filter
  typename ProjectionStackType::RegionType sourceRegion = this->GetInputProjectionStack()->GetLargestPossibleRegion();
  typename ProjectionStackType::SizeType sourceSize = sourceRegion.GetSize();
  typename ProjectionStackType::IndexType sourceIndex = sourceRegion.GetIndex();

  int NumberProjs = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(Dimension-1);
  int FirstProj = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetIndex(Dimension-1);

  // Divide the stack of projections into slabs of projections of identical phase
  std::vector<int> firstProjectionInSlabs;
  std::vector<unsigned int> sizeOfSlabs;
  firstProjectionInSlabs.push_back(FirstProj);
  if (NumberProjs==1)
    sizeOfSlabs.push_back(1);
  else
    {
    for (int proj = FirstProj+1; proj < FirstProj+NumberProjs; proj++)
      {
      if (fabs(m_Signal[proj] - m_Signal[proj-1]) > 1e-4)
        {
        // Compute the number of projections in the current slab
        sizeOfSlabs.push_back(proj - firstProjectionInSlabs[firstProjectionInSlabs.size() - 1]);

        // Update the index of the first projection in the next slab
        firstProjectionInSlabs.push_back(proj);
        }
      }
    sizeOfSlabs.push_back(NumberProjs - firstProjectionInSlabs[firstProjectionInSlabs.size() - 1]);
    }

  bool firstSlabProcessed = false;
  typename VolumeSeriesType::Pointer pimg;

  // Process the projections in order
  for (unsigned int slab = 0; slab < firstProjectionInSlabs.size(); slab++)
    {
    // Set the projection stack source
    sourceIndex[Dimension - 1] = firstProjectionInSlabs[slab];
    sourceSize[Dimension - 1] = sizeOfSlabs[slab];
    this->m_ConstantProjectionStackSource->SetIndex( sourceIndex );
    this->m_ConstantProjectionStackSource->SetSize( sourceSize );

    // Set the interpolation filters, including those for the DVFs
    this->m_InterpolationFilter->SetProjectionNumber(firstProjectionInSlabs[slab]);
    this->m_SplatFilter->SetProjectionNumber(firstProjectionInSlabs[slab]);
    m_DVFInterpolatorFilter->SetFrame(firstProjectionInSlabs[slab]);
    m_InverseDVFInterpolatorFilter->SetFrame(firstProjectionInSlabs[slab]);

    // After the first update, we need to use the output as input.
    if(firstSlabProcessed)
      {
      pimg = this->m_SplatFilter->GetOutput();
      pimg->DisconnectPipeline();
      this->m_SplatFilter->SetInputVolumeSeries( pimg );
      }

    // Update the last filter
    this->m_SplatFilter->Update();

    // Update condition
    firstSlabProcessed = true;
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
