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
#ifndef rtkSubSelectImageFilter_hxx
#define rtkSubSelectImageFilter_hxx

#include "rtkSubSelectImageFilter.h"

namespace rtk
{

template<typename ProjectionStackType>
SubSelectImageFilter<ProjectionStackType>
::SubSelectImageFilter():
  m_OutputGeometry(GeometryType::New()),
  m_EmptyProjectionStackSource(EmptyProjectionStackSourceType::New()),
  m_ExtractFilter(ExtractFilterType::New()),
  m_PasteFilter(PasteFilterType::New())
{
}

template<typename ProjectionStackType>
void SubSelectImageFilter<ProjectionStackType>
::SetInputProjectionStack(const ProjectionStackType* Projections)
{
  this->SetNthInput(0, const_cast<ProjectionStackType*>(Projections));
}

template<typename ProjectionStackType>
typename ProjectionStackType::ConstPointer
SubSelectImageFilter<ProjectionStackType>
::GetInputProjectionStack()
{
  return static_cast< const ProjectionStackType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template<typename ProjectionStackType>
void SubSelectImageFilter<ProjectionStackType>
::GenerateInputRequestedRegion()
{
  const unsigned int Dimension = this->InputImageDimension;

  // Find first selected projection (if it exists)
  unsigned int firstSel = 0;
  for(firstSel = 0; firstSel<m_SelectedProjections.size() && !(m_SelectedProjections[firstSel]); firstSel++);
  if( firstSel == m_SelectedProjections.size() )
    {
    itkGenericExceptionMacro(<< "No projection selected.");
    }

  // Only request the first projection at first
  typename ExtractFilterType::InputImageRegionType projRegion;
  projRegion = this->GetOutput()->GetRequestedRegion();
  projRegion.SetSize(Dimension-1, 1);
  projRegion.SetIndex(Dimension-1, firstSel);
  m_ExtractFilter->SetExtractionRegion(projRegion);
  m_ExtractFilter->UpdateOutputInformation();
  m_ExtractFilter->GetOutput()->SetRequestedRegion(projRegion);
  m_ExtractFilter->GetOutput()->PropagateRequestedRegion();
}

template<typename ProjectionStackType>
void SubSelectImageFilter<ProjectionStackType>
::GenerateOutputInformation()
{
  unsigned int Dimension = this->GetInput(0)->GetImageDimension();
  typename ProjectionStackType::RegionType outputLargestPossibleRegion = this->GetInput(0)->GetLargestPossibleRegion();
  outputLargestPossibleRegion.SetSize(Dimension-1, m_NbSelectedProjs);

  // Create a stack of empty projection images
  typename ProjectionStackType::SizeType ProjectionStackSize;
  ProjectionStackSize = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize();
  ProjectionStackSize[Dimension-1] = m_NbSelectedProjs;
  m_EmptyProjectionStackSource->SetInformationFromImage(this->GetInputProjectionStack());
  m_EmptyProjectionStackSource->SetSize(ProjectionStackSize);
  m_EmptyProjectionStackSource->UpdateOutputInformation();
  this->GetOutput()->CopyInformation( m_EmptyProjectionStackSource->GetOutput() );

  // Mini-pipeline connections
  m_ExtractFilter->SetInput( this->GetInput() );
  m_PasteFilter->SetSourceImage(m_ExtractFilter->GetOutput());
  m_PasteFilter->SetDestinationImage(m_EmptyProjectionStackSource->GetOutput());

  // Update output geometry
  // NOTE : The output geometry must be computed here, not in the GenerateData(),
  // because downstream forward and backprojection filters will need this geometry
  // to compute their output information and input requested region
  m_OutputGeometry->Clear();
  m_OutputGeometry->SetRadiusCylindricalDetector(m_InputGeometry->GetRadiusCylindricalDetector());

  for(unsigned long i=0; i < m_SelectedProjections.size(); i++)
    {
    if (m_SelectedProjections[i])
      {
      m_OutputGeometry->AddProjectionInRadians( m_InputGeometry->GetSourceToIsocenterDistances()[i],
                                                m_InputGeometry->GetSourceToDetectorDistances()[i],
                                                m_InputGeometry->GetGantryAngles()[i],
                                                m_InputGeometry->GetProjectionOffsetsX()[i],
                                                m_InputGeometry->GetProjectionOffsetsY()[i],
                                                m_InputGeometry->GetOutOfPlaneAngles()[i],
                                                m_InputGeometry->GetInPlaneAngles()[i],
                                                m_InputGeometry->GetSourceOffsetsX()[i],
                                                m_InputGeometry->GetSourceOffsetsY()[i]);
      m_OutputGeometry->SetCollimationOfLastProjection(m_InputGeometry->GetCollimationUInf()[i],
                                                       m_InputGeometry->GetCollimationUSup()[i],
                                                       m_InputGeometry->GetCollimationVInf()[i],
                                                       m_InputGeometry->GetCollimationVSup()[i]);
      }
    }
}

template<typename ProjectionStackType>
typename rtk::ThreeDCircularProjectionGeometry::Pointer
SubSelectImageFilter<ProjectionStackType>
::GetOutputGeometry()
{
  return m_OutputGeometry;
}

template<typename ProjectionStackType>
void SubSelectImageFilter<ProjectionStackType>::GenerateData()
{
  unsigned int Dimension = this->GetInput(0)->GetImageDimension();

  // Set the extract filter
  typename ExtractFilterType::InputImageRegionType projRegion;
  projRegion = this->GetOutput()->GetRequestedRegion();
  projRegion.SetSize(Dimension-1, 1);
  m_ExtractFilter->SetExtractionRegion(projRegion);

  // Count the projections actually used in constructing the output
  int counter=0;

  for(unsigned int i=0; i < m_SelectedProjections.size(); i++)
    {
    if (m_SelectedProjections[i])
      {
      // After the first update, we need to use the output as input.
      if(counter>0)
        {
        typename ProjectionStackType::Pointer pimg = m_PasteFilter->GetOutput();
        pimg->DisconnectPipeline();
        m_PasteFilter->SetDestinationImage( pimg );
        }

      // Set the Extract Filter
      projRegion.SetIndex(Dimension - 1, i);
      m_ExtractFilter->SetExtractionRegion(projRegion);

      // Set the Paste filter
      m_PasteFilter->SetSourceRegion( projRegion );
      typename ProjectionStackType::IndexType DestinationIndex = projRegion.GetIndex();
      DestinationIndex[Dimension-1]=counter;
      m_PasteFilter->SetDestinationIndex(DestinationIndex);

      // Update the filters
      projRegion.SetIndex(Dimension - 1, counter);
      m_PasteFilter->UpdateLargestPossibleRegion();

      counter++;
      }
    }

  this->GraftOutput( m_PasteFilter->GetOutput() );
}

}// end namespace


#endif
