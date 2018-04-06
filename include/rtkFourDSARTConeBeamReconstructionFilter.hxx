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

#ifndef rtkFourDSARTConeBeamReconstructionFilter_hxx
#define rtkFourDSARTConeBeamReconstructionFilter_hxx

#include "rtkFourDSARTConeBeamReconstructionFilter.h"
#include "rtkGeneralPurposeFunctions.h"

#include <algorithm>
#include <itkTimeProbe.h>

namespace rtk
{
template<class VolumeSeriesType, class ProjectionStackType>
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::FourDSARTConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set default parameters
  m_EnforcePositivity = false;
  m_NumberOfIterations = 3;
  m_Lambda = 0.3;
  m_ProjectionsOrderInitialized = false;

  // Create each filter of the composite filter
  m_ExtractFilter = ExtractFilterType::New();
  m_ZeroMultiplyFilter = MultiplyFilterType::New();
  m_SubtractFilter = SubtractFilterType::New();
  m_AddFilter = AddFilterType::New();
  m_AddFilter2 = AddFilterType::New();
  m_MultiplyFilter = MultiplyFilterType::New();
  m_ConstantVolumeSeriesSource = ConstantVolumeSeriesSourceType::New();
  m_FourDToProjectionStackFilter = FourDToProjectionStackFilterType::New();
  m_DisplacedDetectorFilter = DisplacedDetectorFilterType::New();
  m_ProjectionStackToFourDFilter = ProjectionStackToFourDFilterType::New();

  // Create the filters required for correct weighting of the difference
  // projection
  m_ExtractFilterRayBox = ExtractFilterType::New();
  m_RayBoxFilter = RayBoxIntersectionFilterType::New();
  m_DivideFilter = DivideFilterType::New();
  m_ConstantProjectionStackSource = ConstantProjectionStackSourceType::New();

  // Create the filter that enforces positivity
  m_ThresholdFilter = ThresholdFilterType::New();
  
  //Permanent internal connections
  m_ZeroMultiplyFilter->SetInput1( itk::NumericTraits<typename InputImageType::PixelType>::ZeroValue() );
  m_ZeroMultiplyFilter->SetInput2( m_ExtractFilter->GetOutput() );

  m_MultiplyFilter->SetInput1( itk::NumericTraits<typename InputImageType::PixelType>::ZeroValue() );
  m_MultiplyFilter->SetInput2( m_SubtractFilter->GetOutput() );

  m_ExtractFilterRayBox->SetInput(m_ConstantProjectionStackSource->GetOutput());
  m_RayBoxFilter->SetInput(m_ExtractFilterRayBox->GetOutput());
  m_DivideFilter->SetInput1(m_MultiplyFilter->GetOutput());
  m_DivideFilter->SetInput2(m_RayBoxFilter->GetOutput());
  m_DisplacedDetectorFilter->SetInput(m_DivideFilter->GetOutput());

  // Default parameters
  m_ExtractFilter->SetDirectionCollapseToSubmatrix();
  m_ExtractFilterRayBox->SetDirectionCollapseToSubmatrix();
  m_NumberOfProjectionsPerSubset = 1; //Default is the SART behavior
  m_DisplacedDetectorFilter->SetPadOnTruncatedSide(false);
  m_DisableDisplacedDetectorFilter = false;
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
  this->SetNthInput(0, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetInputProjectionStack(const ProjectionStackType* Projection)
{
  this->SetNthInput(1, const_cast<ProjectionStackType*>(Projection));
}

template<class VolumeSeriesType, class ProjectionStackType>
typename VolumeSeriesType::ConstPointer
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetInputVolumeSeries()
{
  return static_cast< const VolumeSeriesType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template<class VolumeSeriesType, class ProjectionStackType>
typename ProjectionStackType::Pointer
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GetInputProjectionStack()
{
  return static_cast< ProjectionStackType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetForwardProjectionFilter (int _arg)
{
  if( _arg != this->GetForwardProjectionFilter() )
    {
    Superclass::SetForwardProjectionFilter( _arg );
    m_ForwardProjectionFilter = this->InstantiateForwardProjectionFilter( _arg );
    m_FourDToProjectionStackFilter->SetForwardProjectionFilter( m_ForwardProjectionFilter );
    }
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetBackProjectionFilter (int _arg)
{
  if( _arg != this->GetBackProjectionFilter() )
    {
    Superclass::SetBackProjectionFilter( _arg );
    m_BackProjectionFilter = this->InstantiateBackProjectionFilter( _arg );
    m_ProjectionStackToFourDFilter->SetBackProjectionFilter( m_BackProjectionFilter );
    }
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetWeights(const itk::Array2D<float> _arg)
{
  m_ProjectionStackToFourDFilter->SetWeights(_arg);
  m_FourDToProjectionStackFilter->SetWeights(_arg);
  this->Modified();
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetSignal(const std::vector<double> signal)
{
  m_ProjectionStackToFourDFilter->SetSignal(signal);
  m_FourDToProjectionStackFilter->SetSignal(signal);
  this->m_Signal = signal;
  this->Modified();
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateInputRequestedRegion()
{  
  typename Superclass::InputImagePointer inputPtr =
    const_cast< VolumeSeriesType * >( this->GetInput() );

  if ( !inputPtr )
    return;

  if(m_EnforcePositivity)
    {
    m_ThresholdFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion() );
    m_ThresholdFilter->GetOutput()->PropagateRequestedRegion();
    }
  else
    {
    m_AddFilter2->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion() );
    m_AddFilter2->GetOutput()->PropagateRequestedRegion();
    }
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  const unsigned int Dimension = ProjectionStackType::ImageDimension;
  unsigned int numberOfProjections = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(Dimension-1);

  m_DisplacedDetectorFilter->SetDisable(m_DisableDisplacedDetectorFilter);

  if(!m_ProjectionsOrderInitialized)
    {
    // Fill and shuffle randomly the projection order.
    // Should be tunable with other solutions.
    m_ProjectionsOrder.clear();
    for(unsigned int i = 0; i < numberOfProjections; i++)
      {
      m_ProjectionsOrder.push_back(i);
      }

    std::random_shuffle( m_ProjectionsOrder.begin(), m_ProjectionsOrder.end() );
    m_ProjectionsOrderInitialized = true;
    }

  // We only set the first sub-stack at that point, the rest will be
  // requested in the GenerateData function
  typename ProjectionStackType::RegionType projRegion;
  projRegion = this->GetInputProjectionStack()->GetLargestPossibleRegion();
  projRegion.SetSize(Dimension-1,1);
  projRegion.SetIndex(Dimension-1,m_ProjectionsOrder[0]);
  m_ExtractFilter->SetExtractionRegion(projRegion);
  m_ExtractFilterRayBox->SetExtractionRegion(projRegion);

  // Links with the forward and back projection filters should be set here
  // and not in the constructor, as these filters are set at runtime
  m_ConstantVolumeSeriesSource->SetInformationFromImage(const_cast<VolumeSeriesType *>(this->GetInput(0)));
  m_ConstantVolumeSeriesSource->SetConstant(0);
  m_ConstantVolumeSeriesSource->UpdateOutputInformation();

  m_ProjectionStackToFourDFilter->SetInputVolumeSeries( this->GetInputVolumeSeries() );
  m_ProjectionStackToFourDFilter->SetInputProjectionStack( m_DisplacedDetectorFilter->GetOutput() );
  m_ProjectionStackToFourDFilter->SetSignal(this->m_Signal);

  m_AddFilter->SetInput1(m_ProjectionStackToFourDFilter->GetOutput());
  m_AddFilter->SetInput2(m_ConstantVolumeSeriesSource->GetOutput());

  m_AddFilter2->SetInput1(m_AddFilter->GetOutput());
  m_AddFilter2->SetInput2(this->GetInputVolumeSeries());

  m_ExtractFilter->SetInput( this->GetInputProjectionStack() );

  m_FourDToProjectionStackFilter->SetInputProjectionStack( m_ZeroMultiplyFilter->GetOutput() );
  m_FourDToProjectionStackFilter->SetInputVolumeSeries( this->GetInputVolumeSeries() );

  m_SubtractFilter->SetInput2(m_FourDToProjectionStackFilter->GetOutput() );
  m_SubtractFilter->SetInput1(m_ExtractFilter->GetOutput() );

  // For the same reason, set geometry now
  // Check and set geometry
  if(this->GetGeometry().GetPointer() == ITK_NULLPTR)
    {
    itkGenericExceptionMacro(<< "The geometry of the reconstruction has not been set");
    }
  m_FourDToProjectionStackFilter->SetGeometry(this->m_Geometry);
  m_ProjectionStackToFourDFilter->SetGeometry(this->m_Geometry.GetPointer());
  m_DisplacedDetectorFilter->SetGeometry(this->m_Geometry);

  m_ConstantProjectionStackSource->SetInformationFromImage(const_cast<ProjectionStackType *>(this->GetInputProjectionStack().GetPointer()));
  m_ConstantProjectionStackSource->SetConstant(0);
  m_ConstantProjectionStackSource->UpdateOutputInformation();

  // Create the m_RayBoxFiltersectionImageFilter
  m_RayBoxFilter->SetGeometry(this->GetGeometry().GetPointer());
  itk::Vector<double, 3> Corner1, Corner2;

  Corner1[0] = this->GetInput(0)->GetOrigin()[0];
  Corner1[1] = this->GetInput(0)->GetOrigin()[1];
  Corner1[2] = this->GetInput(0)->GetOrigin()[2];
  Corner2[0] = this->GetInput(0)->GetOrigin()[0] + this->GetInput(0)->GetLargestPossibleRegion().GetSize()[0] * this->GetInput(0)->GetSpacing()[0];
  Corner2[1] = this->GetInput(0)->GetOrigin()[1] + this->GetInput(0)->GetLargestPossibleRegion().GetSize()[1] * this->GetInput(0)->GetSpacing()[1];
  Corner2[2] = this->GetInput(0)->GetOrigin()[2] + this->GetInput(0)->GetLargestPossibleRegion().GetSize()[2] * this->GetInput(0)->GetSpacing()[2];

  m_RayBoxFilter->SetBoxMin(Corner1);
  m_RayBoxFilter->SetBoxMax(Corner2);

  m_RayBoxFilter->UpdateOutputInformation();
  m_ExtractFilter->UpdateOutputInformation();
  m_ZeroMultiplyFilter->UpdateOutputInformation();
  m_FourDToProjectionStackFilter->UpdateOutputInformation();

  m_DivideFilter->UpdateOutputInformation();

  
  if(m_EnforcePositivity)
    {
    m_ThresholdFilter->SetOutsideValue(0);
    m_ThresholdFilter->ThresholdBelow(0);
    m_ThresholdFilter->SetInput(m_AddFilter2->GetOutput() );

    // Update output information
    m_ThresholdFilter->UpdateOutputInformation();
    this->GetOutput()->SetOrigin( m_ThresholdFilter->GetOutput()->GetOrigin() );
    this->GetOutput()->SetSpacing( m_ThresholdFilter->GetOutput()->GetSpacing() );
    this->GetOutput()->SetDirection( m_ThresholdFilter->GetOutput()->GetDirection() );
    this->GetOutput()->SetLargestPossibleRegion( m_ThresholdFilter->GetOutput()->GetLargestPossibleRegion() );
    }
  else
    {
    // Update output information
    m_AddFilter2->UpdateOutputInformation();
    this->GetOutput()->SetOrigin( m_AddFilter2->GetOutput()->GetOrigin() );
    this->GetOutput()->SetSpacing( m_AddFilter2->GetOutput()->GetSpacing() );
    this->GetOutput()->SetDirection( m_AddFilter2->GetOutput()->GetDirection() );
    this->GetOutput()->SetLargestPossibleRegion( m_AddFilter2->GetOutput()->GetLargestPossibleRegion() );
    }

  // Set memory management flags
  m_ZeroMultiplyFilter->ReleaseDataFlagOn();
  m_FourDToProjectionStackFilter->ReleaseDataFlagOn();
  m_SubtractFilter->ReleaseDataFlagOn();
  m_MultiplyFilter->ReleaseDataFlagOn();
  m_RayBoxFilter->ReleaseDataFlagOn();
  m_DivideFilter->ReleaseDataFlagOn();
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateData()
{
  const unsigned int Dimension = ProjectionStackType::ImageDimension;

  typename ProjectionStackType::RegionType subsetRegion;
  subsetRegion = this->GetInputProjectionStack()->GetLargestPossibleRegion();
  unsigned int nProj = subsetRegion.GetSize(Dimension-1);
  subsetRegion.SetSize(Dimension-1, 1);

  m_MultiplyFilter->SetInput1( (const float) m_Lambda/(double)m_NumberOfProjectionsPerSubset  );
  
  // Create the zero projection stack used as input by RayBoxIntersectionFilter
  m_ConstantProjectionStackSource->Update();

  // Declare the image used in the main loop
  typename VolumeSeriesType::Pointer pimg;
  typename VolumeSeriesType::Pointer pimg2;

  // For each iteration, go over each projection
  for(unsigned int iter = 0; iter < m_NumberOfIterations; iter++)
    {
    unsigned int projectionsProcessedInSubset = 0;

    for(unsigned int i = 0; i < nProj; i++)
      {
      // When we reach the number of projections per subset:
      // - plug the output of the pipeline back into the Forward projection filter
      // - set the input of the Back projection filter to zero
      // - reset the projectionsProcessedInSubset to zero
      if (projectionsProcessedInSubset == m_NumberOfProjectionsPerSubset)
        {
        if (m_EnforcePositivity)
          pimg2 = m_ThresholdFilter->GetOutput();
        else
          pimg2 = m_AddFilter2->GetOutput();

        pimg2->DisconnectPipeline();

        m_FourDToProjectionStackFilter->SetInputVolumeSeries( pimg2 );
        m_AddFilter2->SetInput2( pimg2 );
        m_AddFilter->SetInput2(m_ConstantVolumeSeriesSource->GetOutput());

        projectionsProcessedInSubset = 0;
        }

      // Otherwise, just plug the output of the add filter
      // back as its input
      else
        {
        if (i)
          {
          pimg = m_AddFilter->GetOutput();
          pimg->DisconnectPipeline();
          m_AddFilter->SetInput2(pimg);
          }
        else
          {
          m_AddFilter->SetInput2(m_ConstantVolumeSeriesSource->GetOutput());
          }
        }

      // Change projection subset
      subsetRegion.SetIndex( Dimension-1, m_ProjectionsOrder[i] );
      m_ExtractFilter->SetExtractionRegion(subsetRegion);
      m_ExtractFilterRayBox->SetExtractionRegion(subsetRegion);

      // This is required to reset the full pipeline
      m_ProjectionStackToFourDFilter->GetOutput()->UpdateOutputInformation();
      m_ProjectionStackToFourDFilter->GetOutput()->PropagateRequestedRegion();

      m_ExtractProbe.Start();
      m_ExtractFilter->Update();
      m_ExtractFilterRayBox->Update();
      m_ExtractProbe.Stop();

      m_ZeroMultiplyProbe.Start();
      m_ZeroMultiplyFilter->Update();
      m_ZeroMultiplyProbe.Stop();

      m_ForwardProjectionProbe.Start();
      m_FourDToProjectionStackFilter->Update();
      m_ForwardProjectionProbe.Stop();

      m_SubtractProbe.Start();
      m_SubtractFilter->Update();
      m_SubtractProbe.Stop();

      m_MultiplyProbe.Start();
      m_MultiplyFilter->Update();
      m_MultiplyProbe.Stop();

      m_RayBoxProbe.Start();
      m_RayBoxFilter->Update();
      m_RayBoxProbe.Stop();

      m_DivideProbe.Start();
      m_DivideFilter->Update();
      m_DivideProbe.Stop();

      m_DisplacedDetectorProbe.Start();
      m_DisplacedDetectorFilter->Update();
      m_DisplacedDetectorProbe.Stop();

      m_BackProjectionProbe.Start();
      m_ProjectionStackToFourDFilter->Update();
      m_BackProjectionProbe.Stop();

      m_AddProbe.Start();
      m_AddFilter->Update();
      m_AddProbe.Stop();

      projectionsProcessedInSubset++;
      if ((projectionsProcessedInSubset == m_NumberOfProjectionsPerSubset) || (i == nProj - 1))
        {
        m_AddProbe.Start();
        m_AddFilter2->SetInput1(m_AddFilter->GetOutput());
        m_AddFilter2->Update();
        m_AddProbe.Stop();

        if (m_EnforcePositivity)
          {
          m_ThresholdProbe.Start();
          m_ThresholdFilter->Update();
          m_ThresholdProbe.Stop();
          }
        }

      }
    }
  if (m_EnforcePositivity)
    {
    this->GraftOutput( m_ThresholdFilter->GetOutput() );
    }
  else
    {
    this->GraftOutput( m_AddFilter2->GetOutput() );
    }
}

template<class VolumeSeriesType, class ProjectionStackType>
void
FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::PrintTiming(std::ostream & os) const
{
  os << "FourDSARTConeBeamReconstructionFilter timing:" << std::endl;
  os << "  Extraction of projection sub-stacks: " << m_ExtractProbe.GetTotal()
     << ' ' << m_ExtractProbe.GetUnit() << std::endl;
  os << "  Multiplication by zero: " << m_ZeroMultiplyProbe.GetTotal()
     << ' ' << m_ZeroMultiplyProbe.GetUnit() << std::endl;
  os << "  Forward projection: " << m_ForwardProjectionProbe.GetTotal()
     << ' ' << m_ForwardProjectionProbe.GetUnit() << std::endl;
  os << "  Subtraction: " << m_SubtractProbe.GetTotal()
     << ' ' << m_SubtractProbe.GetUnit() << std::endl;
  os << "  Multiplication by lambda: " << m_MultiplyProbe.GetTotal()
     << ' ' << m_MultiplyProbe.GetUnit() << std::endl;
  os << "  Ray box intersection: " << m_RayBoxProbe.GetTotal()
     << ' ' << m_RayBoxProbe.GetUnit() << std::endl;
  os << "  Division: " << m_DivideProbe.GetTotal()
     << ' ' << m_DivideProbe.GetUnit() << std::endl;
  os << "  Displaced detector: " << m_DisplacedDetectorProbe.GetTotal()
     << ' ' << m_DisplacedDetectorProbe.GetUnit() << std::endl;
  os << "  Back projection: " << m_BackProjectionProbe.GetTotal()
     << ' ' << m_BackProjectionProbe.GetUnit() << std::endl;
  os << "  Volume update: " << m_AddProbe.GetTotal()
     << ' ' << m_AddProbe.GetUnit() << std::endl;
  if (m_EnforcePositivity)
    {
    os << "  Positivity enforcement: " << m_ThresholdProbe.GetTotal()
    << ' ' << m_ThresholdProbe.GetUnit() << std::endl;
    }
}

} // end namespace rtk

#endif // rtkFourDSARTConeBeamReconstructionFilter_hxx
