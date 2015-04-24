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
#ifndef __rtkProjectionStackToFourDImageFilter_txx
#define __rtkProjectionStackToFourDImageFilter_txx

#include "rtkProjectionStackToFourDImageFilter.h"

#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

namespace rtk
{

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>::ProjectionStackToFourDImageFilter()
{
  this->SetNumberOfRequiredInputs(2);

  m_ProjectionNumber = 0;
  m_UseCudaSplat = false;
  m_UseCudaSources = false;

  // Create the filters
  m_ExtractFilter = ExtractFilterType::New();
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
  this->SetNthInput(0, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>::SetInputProjectionStack(const ProjectionStackType* Projection)
{
  this->SetNthInput(1, const_cast<ProjectionStackType*>(Projection));
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
typename VolumeSeriesType::ConstPointer ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>::GetInputVolumeSeries()
{
  return static_cast< const VolumeSeriesType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
typename ProjectionStackType::ConstPointer ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>::GetInputProjectionStack()
{
  return static_cast< const ProjectionStackType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg)
{
  m_BackProjectionFilter = _arg;
  this->Modified();
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg)
{
  m_Geometry = _arg;
  this->Modified();
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::InitializeConstantSource()
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

  m_ConstantVolumeSource->SetOrigin( ConstantVolumeSourceOrigin );
  m_ConstantVolumeSource->SetSpacing( ConstantVolumeSourceSpacing );
  m_ConstantVolumeSource->SetDirection( ConstantVolumeSourceDirection );
  m_ConstantVolumeSource->SetSize( ConstantVolumeSourceSize );
  m_ConstantVolumeSource->SetConstant( 0. );

  // Configure the constant volume series source
  m_ConstantVolumeSeriesSource->SetInformationFromImage(this->GetInputVolumeSeries());
  m_ConstantVolumeSeriesSource->SetConstant( 0. );
  m_ConstantVolumeSeriesSource->ReleaseDataFlagOn();
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::GenerateOutputInformation()
{
  // Create and set the splat filter
  m_SplatFilter = SplatFilterType::New();
#ifdef RTK_USE_CUDA
  if (m_UseCudaSplat)
    m_SplatFilter = rtk::CudaSplatImageFilter::New();
#endif
  
  // Create the constant sources (first on CPU, and overwrite with the GPU version if CUDA requested)
  m_ConstantVolumeSource = ConstantVolumeSourceType::New();
  m_ConstantVolumeSeriesSource = ConstantVolumeSeriesSourceType::New();
  m_DisplacedDetectorFilter = DisplacedDetectorFilterType::New();
#ifdef RTK_USE_CUDA
  if (m_UseCudaSources)
    {
    m_ConstantVolumeSource = rtk::CudaConstantVolumeSource::New();
    m_ConstantVolumeSeriesSource = rtk::CudaConstantVolumeSeriesSource::New();
    }
  m_DisplacedDetectorFilter = rtk::CudaDisplacedDetectorImageFilter::New();
#endif

  // Set runtime connections
  m_ExtractFilter->SetInput(this->GetInputProjectionStack());

  m_DisplacedDetectorFilter->SetInput(m_ExtractFilter->GetOutput());

  m_BackProjectionFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_DisplacedDetectorFilter->GetOutput());
  m_BackProjectionFilter->SetInPlace(false);

  m_SplatFilter->SetInputVolumeSeries(m_ConstantVolumeSeriesSource->GetOutput());
  m_SplatFilter->SetInputVolume(m_BackProjectionFilter->GetOutput());

  // Set runtime parameters
  m_ProjectionNumber = 0;
  m_BackProjectionFilter->SetGeometry(m_Geometry.GetPointer());
  m_DisplacedDetectorFilter->SetGeometry(m_Geometry);
  m_SplatFilter->SetProjectionNumber(m_ProjectionNumber);
  m_SplatFilter->SetWeights(m_Weights);

  // Prepare the extract filter
  int Dimension = ProjectionStackType::ImageDimension; // Dimension=3
  unsigned int NumberProjs = GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(2);
  if (NumberProjs != m_Weights.columns())
    {
    std::cerr << "Size of interpolation weights array does not match the number of projections" << std::endl;
    }

  typename ExtractFilterType::InputImageRegionType subsetRegion;
  subsetRegion = GetInputProjectionStack()->GetLargestPossibleRegion();
  subsetRegion.SetSize(Dimension-1, 1);
  subsetRegion.SetIndex(Dimension-1, m_ProjectionNumber);
  m_ExtractFilter->SetExtractionRegion(subsetRegion);

  // Have the last filter calculate its output information
  this->InitializeConstantSource();
  m_SplatFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation(m_SplatFilter->GetOutput());
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::GenerateInputRequestedRegion()
{
  // Input 0 is the volume series we update
  typename VolumeSeriesType::Pointer  inputPtr0 = const_cast< VolumeSeriesType * >( this->GetInput(0) );
  if ( !inputPtr0 )
    {
    return;
    }
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the stack of projections
  typename ProjectionStackType::Pointer inputPtr1 = static_cast< ProjectionStackType * >
            ( this->itk::ProcessObject::GetInput(1) );
  inputPtr1->SetRequestedRegion(this->m_BackProjectionFilter->GetInput(1)->GetRequestedRegion());
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::GenerateData()
{
  int Dimension = ProjectionStackType::ImageDimension;

  // Set the Extract filter
  typename ProjectionStackType::RegionType extractRegion;
  extractRegion = this->GetInputProjectionStack()->GetLargestPossibleRegion();
  extractRegion.SetSize(Dimension-1, 1);

  // Declare the pointer to a VolumeSeries that will be used in the pipeline
  typename VolumeSeriesType::Pointer pimg;

  int NumberProjs = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(2);
  for(m_ProjectionNumber=0; m_ProjectionNumber<NumberProjs; m_ProjectionNumber++)
    {
    // After the first update, we need to use the output as input.
    if(m_ProjectionNumber>0)
      {
      pimg = m_SplatFilter->GetOutput();
      pimg->DisconnectPipeline();
      m_SplatFilter->SetInputVolumeSeries( pimg );
      }

    // Set the Extract Filter
    extractRegion.SetIndex(Dimension-1, m_ProjectionNumber);
    m_ExtractFilter->SetExtractionRegion(extractRegion);

    // Set the splat filter
    m_SplatFilter->SetProjectionNumber(m_ProjectionNumber);

    // Update the last filter
    m_SplatFilter->Update();
    }

  // Graft its output
  this->GraftOutput( m_SplatFilter->GetOutput() );

  // Release the data in internal filters
  pimg->ReleaseData();
  m_DisplacedDetectorFilter->GetOutput()->ReleaseData();
  m_BackProjectionFilter->GetOutput()->ReleaseData();
  m_ExtractFilter->GetOutput()->ReleaseData();
  m_ConstantVolumeSource->GetOutput()->ReleaseData();
}

}// end namespace


#endif
