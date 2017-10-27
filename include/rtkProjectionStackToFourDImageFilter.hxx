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
#ifndef rtkProjectionStackToFourDImageFilter_hxx
#define rtkProjectionStackToFourDImageFilter_hxx

#include "rtkProjectionStackToFourDImageFilter.h"
#include "rtkGeneralPurposeFunctions.h"

#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

namespace rtk
{

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>::ProjectionStackToFourDImageFilter()
{
  this->SetNumberOfRequiredInputs(2);

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
ProjectionStackToFourDImageFilter< VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::SetSignal(const std::vector<double> signal)
{
  this->m_Signal = signal;
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
#ifdef RTK_USE_CUDA
  if (m_UseCudaSources)
    {
    m_ConstantVolumeSource = rtk::CudaConstantVolumeSource::New();
    m_ConstantVolumeSeriesSource = rtk::CudaConstantVolumeSeriesSource::New();
    }
#endif

  // Set runtime connections
  m_ExtractFilter->SetInput(this->GetInputProjectionStack());

  m_BackProjectionFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_ExtractFilter->GetOutput());
  m_BackProjectionFilter->SetInPlace(false);

  m_SplatFilter->SetInputVolumeSeries(m_ConstantVolumeSeriesSource->GetOutput());
  m_SplatFilter->SetInputVolume(m_BackProjectionFilter->GetOutput());

  // Prepare the extract filter
  int Dimension = ProjectionStackType::ImageDimension; // Dimension=3
//  unsigned int NumberProjs = GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(2);
//  if (NumberProjs != m_Weights.columns())
//    itkWarningMacro("Size of interpolation weights array does not match the number of projections");

  typename ExtractFilterType::InputImageRegionType subsetRegion;
  subsetRegion = GetInputProjectionStack()->GetLargestPossibleRegion();
  subsetRegion.SetSize(Dimension-1, 1);
  m_ExtractFilter->SetExtractionRegion(subsetRegion);

  // Set runtime parameters
  m_BackProjectionFilter->SetGeometry(m_Geometry.GetPointer());
  m_SplatFilter->SetProjectionNumber(subsetRegion.GetIndex(Dimension-1));
  m_SplatFilter->SetWeights(m_Weights);

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
  // The 4D input volume need not be loaded in memory, is it only used to configure the
  // m_ConstantVolumeSeriesSource with the correct information
  // Leave its requested region unchanged (set by the other filters that need it)

  // Calculation of the requested region on input 1 is left to the back projection filter
  this->m_BackProjectionFilter->PropagateRequestedRegion(this->m_BackProjectionFilter->GetOutput());
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::GenerateData()
{
  int Dimension = ProjectionStackType::ImageDimension;

  // Prepare the index for the constant projection stack source and the extract filter
  typename ProjectionStackType::RegionType extractRegion = this->GetInputProjectionStack()->GetLargestPossibleRegion();
  typename ProjectionStackType::SizeType extractSize = extractRegion.GetSize();
  typename ProjectionStackType::IndexType extractIndex = extractRegion.GetIndex();

  int NumberProjs = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(Dimension-1);
  int FirstProj = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetIndex(Dimension-1);

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
    extractIndex[Dimension - 1] = firstProjectionInSlabs[slab];
    extractSize[Dimension - 1] = sizeOfSlabs[slab];
    extractRegion.SetIndex(extractIndex);
    extractRegion.SetSize(extractSize);
    m_ExtractFilter->SetExtractionRegion(extractRegion);

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
  if(pimg.IsNotNull())
    pimg->ReleaseData();
  m_BackProjectionFilter->GetOutput()->ReleaseData();
  m_ExtractFilter->GetOutput()->ReleaseData();
  m_ConstantVolumeSource->GetOutput()->ReleaseData();
}

}// end namespace


#endif
