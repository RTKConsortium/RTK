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
#ifndef rtkPhaseGatingImageFilter_hxx
#define rtkPhaseGatingImageFilter_hxx

#include "rtkPhaseGatingImageFilter.h"

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>

#include "math.h"

namespace rtk
{

template<typename ProjectionStackType>
PhaseGatingImageFilter<ProjectionStackType>::PhaseGatingImageFilter()
{
  m_PhaseReader = PhaseReader::New();

  // Set default parameters
  m_GatingWindowWidth = 1;
  m_GatingWindowShape = 0;
  m_GatingWindowCenter = 0.5;
}

template<typename ProjectionStackType>
void PhaseGatingImageFilter<ProjectionStackType>::ComputeWeights()
{
  m_GatingWeights.clear();
  float distance;

  // Compute the gating weights
  for(unsigned int proj=0; proj<m_Phases.size(); proj++)
  {
  distance = vnl_math_min(fabs(m_GatingWindowCenter - 1 - m_Phases[proj]), fabs(m_GatingWindowCenter - m_Phases[proj]));
  distance = vnl_math_min(distance, vnl_math_abs(m_GatingWindowCenter + 1.f - m_Phases[proj]));

  switch(m_GatingWindowShape)
    {
    case(0): // Rectangular
      if (2 * distance <= m_GatingWindowWidth) m_GatingWeights.push_back(1);
      else m_GatingWeights.push_back(0);
      break;
    case(1): // Triangular
      m_GatingWeights.push_back(vnl_math_max(1.f - 2.f * distance / m_GatingWindowWidth, 0.f));
      break;
    default:
      std::cerr << "Unhandled gating window shape value." << std::endl;
    }
  }
}

template<typename ProjectionStackType>
void PhaseGatingImageFilter<ProjectionStackType>::GenerateOutputInformation()
{
  m_PhaseReader->SetFileName(m_PhasesFileName);
  m_PhaseReader->SetFieldDelimiterCharacter( ';' );
  m_PhaseReader->HasRowHeadersOff();
  m_PhaseReader->HasColumnHeadersOff();
  m_PhaseReader->Update();
  SetPhases(m_PhaseReader->GetOutput());
  ComputeWeights();
  SelectProjections();

  Superclass::GenerateOutputInformation();
}

template<typename ProjectionStackType>
void PhaseGatingImageFilter<ProjectionStackType>::SetPhases(std::vector<float> phases)
{
  m_Phases = phases;
}

template<typename ProjectionStackType>
std::vector<float>
PhaseGatingImageFilter<ProjectionStackType>::GetGatingWeights()
{
  return m_GatingWeights;
}

template<typename ProjectionStackType>
std::vector<float>
PhaseGatingImageFilter<ProjectionStackType>::GetGatingWeightsOnSelectedProjections()
{
  return m_GatingWeightsOnSelectedProjections;
}

template<typename ProjectionStackType>
void
PhaseGatingImageFilter<ProjectionStackType>::SelectProjections()
{
  float eps=0.0001;
  this->m_SelectedProjections.resize(m_GatingWeights.size(), false);
  this->m_NbSelectedProjs=0;
  m_GatingWeightsOnSelectedProjections.clear();

  // Select only those projections that have non-zero weights to speed up computing
  for (unsigned int i=0; i < m_GatingWeights.size(); i++)
    {
    if (m_GatingWeights[i]>eps)
      {
      this->m_SelectedProjections[i]=true;
      this->m_NbSelectedProjs++;
      m_GatingWeightsOnSelectedProjections.push_back(m_GatingWeights[i]);
      }
    }
}

}// end namespace


#endif
