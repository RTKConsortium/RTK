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
#ifndef rtkFourDReconstructionConjugateGradientOperator_hxx
#define rtkFourDReconstructionConjugateGradientOperator_hxx

#include "rtkFourDReconstructionConjugateGradientOperator.h"
#include "rtkGeneralPurposeFunctions.h"

namespace rtk
{

template< typename VolumeSeriesType, typename ProjectionStackType>
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>::FourDReconstructionConjugateGradientOperator()
{
  this->SetNumberOfRequiredInputs(2);

  // Set default behavior
  m_UseCudaInterpolation = false;
  m_UseCudaSplat = false;
  m_UseCudaSources = false;

#ifdef RTK_USE_CUDA
  m_DisplacedDetectorFilter = rtk::CudaDisplacedDetectorImageFilter::New();
#else
  m_DisplacedDetectorFilter = DisplacedDetectorFilterType::New();
#endif
  m_DisplacedDetectorFilter->SetPadOnTruncatedSide(false);
  m_DisableDisplacedDetectorFilter = false;

  // Set memory management flags
  m_DisplacedDetectorFilter->SetInPlace(true);
  m_DisplacedDetectorFilter->ReleaseDataFlagOn();
}


template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
  this->SetNthInput(0, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>::SetInputProjectionStack(const ProjectionStackType *Projections)
{
  this->SetNthInput(1, const_cast<ProjectionStackType*>(Projections));
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename VolumeSeriesType::ConstPointer
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>::GetInputVolumeSeries()
{
  return static_cast< const VolumeSeriesType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
typename ProjectionStackType::ConstPointer
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>::GetInputProjectionStack()
{
  return static_cast< const ProjectionStackType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
::SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg)
{
  m_BackProjectionFilter = _arg;
  this->Modified();
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
::SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg)
{
  m_ForwardProjectionFilter = _arg;
  this->Modified();
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>
::SetSignal(const std::vector<double> signal)
{
  this->m_Signal = signal;
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
::InitializeConstantSources()
{
  unsigned int Dimension = 3;

  // Configure the constant volume sources
  typename VolumeType::SizeType ConstantVolumeSourceSize;
  typename VolumeType::SpacingType ConstantVolumeSourceSpacing;
  typename VolumeType::PointType ConstantVolumeSourceOrigin;
  typename VolumeType::DirectionType ConstantVolumeSourceDirection;

  ConstantVolumeSourceSize.Fill(0);
  ConstantVolumeSourceSpacing.Fill(0);
  ConstantVolumeSourceOrigin.Fill(0);

  for(unsigned int i=0; i < Dimension; i++)
    {
    ConstantVolumeSourceSize[i] = GetInputVolumeSeries()->GetLargestPossibleRegion().GetSize()[i];
    ConstantVolumeSourceSpacing[i] = GetInputVolumeSeries()->GetSpacing()[i];
    ConstantVolumeSourceOrigin[i] = GetInputVolumeSeries()->GetOrigin()[i];
    }
  ConstantVolumeSourceDirection.SetIdentity();

  m_ConstantVolumeSource1->SetOrigin( ConstantVolumeSourceOrigin );
  m_ConstantVolumeSource1->SetSpacing( ConstantVolumeSourceSpacing );
  m_ConstantVolumeSource1->SetDirection( ConstantVolumeSourceDirection );
  m_ConstantVolumeSource1->SetSize( ConstantVolumeSourceSize );
  m_ConstantVolumeSource1->SetConstant( 0. );

  m_ConstantVolumeSource2->SetOrigin( ConstantVolumeSourceOrigin );
  m_ConstantVolumeSource2->SetSpacing( ConstantVolumeSourceSpacing );
  m_ConstantVolumeSource2->SetDirection( ConstantVolumeSourceDirection );
  m_ConstantVolumeSource2->SetSize( ConstantVolumeSourceSize );
  m_ConstantVolumeSource2->SetConstant( 0. );

  // Configure the constant projection stack source
  m_ConstantProjectionStackSource->SetInformationFromImage(this->GetInputProjectionStack());
  typename ProjectionStackType::SizeType ConstantProjectionStackSourceSize
      = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize();
  ConstantProjectionStackSourceSize[Dimension - 1] = 1;
  m_ConstantProjectionStackSource->SetSize( ConstantProjectionStackSourceSize );
  m_ConstantProjectionStackSource->SetConstant( 0. );

  // Configure the constant volume series source
  m_ConstantVolumeSeriesSource->SetInformationFromImage(this->GetInputVolumeSeries());
  m_ConstantVolumeSeriesSource->SetConstant( 0. );

  // Configure memory management options
  m_ConstantProjectionStackSource->ReleaseDataFlagOn();
  m_ConstantVolumeSeriesSource->ReleaseDataFlagOn();
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  // Set runtime parameters
  m_DisplacedDetectorFilter->SetDisable(m_DisableDisplacedDetectorFilter);

  // Create the interpolation filter and the displaced detector filter
  // (first on CPU, and overwrite with the GPU version if CUDA requested)
  m_InterpolationFilter = InterpolationFilterType::New();
#ifdef RTK_USE_CUDA
  if (m_UseCudaInterpolation)
    m_InterpolationFilter = rtk::CudaInterpolateImageFilter::New();
#endif

  // Create the splat filter (first on CPU, and overwrite with the GPU version if CUDA requested)
  m_SplatFilter = SplatFilterType::New();
#ifdef RTK_USE_CUDA
  if (m_UseCudaSplat)
    m_SplatFilter = rtk::CudaSplatImageFilter::New();
#endif

  // Create the constant sources (first on CPU, and overwrite with the GPU version if CUDA requested)
  m_ConstantVolumeSource1 = ConstantVolumeSourceType::New();
  m_ConstantVolumeSource2 = ConstantVolumeSourceType::New();
  m_ConstantProjectionStackSource = ConstantProjectionStackSourceType::New();
  m_ConstantVolumeSeriesSource = ConstantVolumeSeriesSourceType::New();
#ifdef RTK_USE_CUDA
  if (m_UseCudaSources)
    {
    m_ConstantVolumeSource1 = rtk::CudaConstantVolumeSource::New();
    m_ConstantVolumeSource2 = rtk::CudaConstantVolumeSource::New();
    m_ConstantProjectionStackSource = rtk::CudaConstantVolumeSource::New();
    m_ConstantVolumeSeriesSource = rtk::CudaConstantVolumeSeriesSource::New();
    }
#endif

  // Initialize sources
  this->InitializeConstantSources();

  // Set runtime connections
  m_InterpolationFilter->SetInputVolume(m_ConstantVolumeSource1->GetOutput());
  m_InterpolationFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());

  m_ForwardProjectionFilter->SetInput(0, m_ConstantProjectionStackSource->GetOutput());
  m_ForwardProjectionFilter->SetInput(1, m_InterpolationFilter->GetOutput());

  m_DisplacedDetectorFilter->SetInput(m_ForwardProjectionFilter->GetOutput());

  m_BackProjectionFilter->SetInput(0, m_ConstantVolumeSource2->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_DisplacedDetectorFilter->GetOutput());
  m_BackProjectionFilter->SetInPlace(false);

  m_SplatFilter->SetInputVolumeSeries(m_ConstantVolumeSeriesSource->GetOutput());
  m_SplatFilter->SetInputVolume(m_BackProjectionFilter->GetOutput());

  m_InterpolationFilter->SetWeights(m_Weights);
  m_SplatFilter->SetWeights(m_Weights);
  m_InterpolationFilter->SetProjectionNumber(0);
  m_SplatFilter->SetProjectionNumber(0);

  // Set geometry
  m_BackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());
  m_ForwardProjectionFilter->SetGeometry(this->m_Geometry);
  m_DisplacedDetectorFilter->SetGeometry(this->m_Geometry);

  // Have the last filter calculate its output information
  m_SplatFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_SplatFilter->GetOutput() );
}


template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
::GenerateInputRequestedRegion()
{
  // Let the internal filters compute the input requested region
  m_SplatFilter->PropagateRequestedRegion(m_SplatFilter->GetOutput());
  this->m_ForwardProjectionFilter->PropagateRequestedRegion(this->m_ForwardProjectionFilter->GetOutput());
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
::GenerateData()
{
  int Dimension = ProjectionStackType::ImageDimension;

  // Prepare the index for the constant projection stack source
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

    // Set the Interpolation filter
    m_InterpolationFilter->SetProjectionNumber(firstProjectionInSlabs[slab]);
    m_SplatFilter->SetProjectionNumber(firstProjectionInSlabs[slab]);

    // After the first update, we need to use the output as input.
    if(firstSlabProcessed)
      {
      pimg = this->m_SplatFilter->GetOutput();
      pimg->DisconnectPipeline();
      this->m_SplatFilter->SetInputVolumeSeries( pimg );
      }

    // Update the last filter
    m_SplatFilter->Update();

    // Update condition
    firstSlabProcessed = true;
    }

  // Graft its output
  this->GraftOutput( m_SplatFilter->GetOutput() );

  // Release the data in internal filters
  pimg->ReleaseData();
  m_ConstantVolumeSource1->GetOutput()->ReleaseData();
  m_ConstantVolumeSource2->GetOutput()->ReleaseData();
  m_ConstantVolumeSeriesSource->GetOutput()->ReleaseData();
  m_ConstantProjectionStackSource->GetOutput()->ReleaseData();
  m_DisplacedDetectorFilter->GetOutput()->ReleaseData();
  m_InterpolationFilter->GetOutput()->ReleaseData();
  m_BackProjectionFilter->GetOutput()->ReleaseData();
  m_ForwardProjectionFilter->GetOutput()->ReleaseData();

  // Send the input back onto the CPU
  this->GetInputVolumeSeries()->GetBufferPointer();
}

}// end namespace


#endif
