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

#ifndef __rtkIncrementalFourDConjugateGradientConeBeamReconstructionFilter_hxx
#define __rtkIncrementalFourDConjugateGradientConeBeamReconstructionFilter_hxx

#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkIncrementalFourDConjugateGradientConeBeamReconstructionFilter.h"

#include <algorithm>
#include <itkImageFileWriter.h>

namespace rtk
{

template<class VolumeSeriesType, class ProjectionStackType>
IncrementalFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::IncrementalFourDConjugateGradientConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set the default values of member parameters
  m_MainLoop_iterations=3;
  m_CG_iterations=2;
  m_NumberOfProjectionsPerSubset=0;
  m_NumberOfSubsets=1;
  m_CudaConjugateGradient = false; // 4D volumes of usual size only fit on the largest GPUs

  // Create the filters
  m_CG = CGType::New();
}

template<class VolumeSeriesType, class ProjectionStackType>
void
IncrementalFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
  this->SetNthInput(0, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template<class VolumeSeriesType, class ProjectionStackType>
void
IncrementalFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::SetInputProjectionStack(const ProjectionStackType* Projection)
{
  this->SetNthInput(1, const_cast<ProjectionStackType*>(Projection));
}

template<class VolumeSeriesType, class ProjectionStackType>
void
IncrementalFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::SetInputProjectionWeights(const ProjectionStackType* ProjectionWeights)
{
  this->SetInput("ProjectionWeights", const_cast<ProjectionStackType*>(ProjectionWeights));
}

template<class VolumeSeriesType, class ProjectionStackType>
typename VolumeSeriesType::ConstPointer
IncrementalFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::GetInputVolumeSeries()
{
  return static_cast< const VolumeSeriesType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template<class VolumeSeriesType, class ProjectionStackType>
typename ProjectionStackType::ConstPointer
IncrementalFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::GetInputProjectionStack()
{
  return static_cast< const VolumeType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template<class VolumeSeriesType, class ProjectionStackType>
typename ProjectionStackType::ConstPointer
IncrementalFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::GetInputProjectionWeights()
{
  return static_cast< const ProjectionStackType * >
          ( this->itk::ProcessObject::GetInput("ProjectionWeights") );
}

template<class VolumeSeriesType, class ProjectionStackType>
void
IncrementalFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetForwardProjectionFilter (int _arg)
{
  m_CG->SetForwardProjectionFilter(_arg);
  this->Modified();
}


template<class VolumeSeriesType, class ProjectionStackType>
void
IncrementalFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetBackProjectionFilter (int _arg)
{
  m_CG->SetBackProjectionFilter(_arg);
  this->Modified();
}

template<class VolumeSeriesType, class ProjectionStackType>
void
IncrementalFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::SetSignal(const std::vector<double> signal)
{
  m_CG->SetSignal(signal);
  this->m_Signal = signal;
  this->Modified();
}

template<class VolumeSeriesType, class ProjectionStackType>
void
IncrementalFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  unsigned int Dimension = ProjectionStackType::ImageDimension;

  // Divide the set of projections into subsets, each one containing nprojpersubset projections (except maybe the last subset)
  // For each subset, store the selectedProjs vector, and create one SubSelectFromListImageFilter and one PhasesToInterpolationWeights filter
  unsigned int nProjs = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(Dimension - 1);

  // If the number of projections per subset has not been set, use the whole projection stack every time
  if (m_NumberOfProjectionsPerSubset == 0)
    {
    m_NumberOfSubsets = 1;
    m_NumberOfProjectionsPerSubset = nProjs;
    }
  // Otherwise compute the number of subsets
  else
    m_NumberOfSubsets = ceil((float)nProjs / (float) m_NumberOfProjectionsPerSubset);

  // Create a vector with the projection indices and shuffle it,
  // in order to build the subsets with randomly chosen projections
  std::vector< unsigned int > projOrder(nProjs);
  for(unsigned int i = 0; i < nProjs; i++)
    projOrder[i] = i;
  std::random_shuffle( projOrder.begin(), projOrder.end() );

  for (unsigned int subset = 0; subset < m_NumberOfSubsets; subset+=2)
    {
    // Create a vector of booleans, indicating whether a projection is selected or not
    std::vector< bool > selectedProjs(nProjs);
    std::fill(selectedProjs.begin(), selectedProjs.end(), false);
    for(unsigned int proj = subset * m_NumberOfProjectionsPerSubset; (proj < nProjs) && (proj < (subset + 1) * m_NumberOfProjectionsPerSubset); proj++)
      selectedProjs[projOrder[proj]] = true;

    // Store this vector
    m_SelectedProjsVector.push_back(selectedProjs);

    // Use the SubSelectFromListImageFilter to extract the substack of projections and the subgeometry
    m_SubSelectFilters.push_back(SubSelectType::New());
    m_SubSelectFilters[subset]->SetSelectedProjections( m_SelectedProjsVector[subset] );
    m_SubSelectFilters[subset]->SetInputGeometry( m_Geometry );
    m_SubSelectFilters[subset]->SetInputProjectionStack( this->GetInputProjectionStack() );
    m_SubSelectFilters[subset]->Update();

    // Do the same for the projection weights (except that the geometry is useless)
    m_SubSelectFilters.push_back(SubSelectType::New());
    m_SubSelectFilters[subset+1]->SetSelectedProjections( m_SelectedProjsVector[subset] );
    m_SubSelectFilters[subset+1]->SetInputGeometry( m_Geometry );
    m_SubSelectFilters[subset+1]->SetInputProjectionStack( this->GetInputProjectionWeights() );
    m_SubSelectFilters[subset+1]->Update();

    // Read the phases file, and extract the interpolation weights for this subset
    m_PhasesFilters.push_back(rtk::PhasesToInterpolationWeights::New());
    m_PhasesFilters[subset]->SetFileName(m_PhasesFileName);
    m_PhasesFilters[subset]->SetNumberOfReconstructedFrames(this->GetInputVolumeSeries()->GetLargestPossibleRegion().GetSize(Dimension));
    m_PhasesFilters[subset]->SetSelectedProjections( m_SelectedProjsVector[subset] );
    m_PhasesFilters[subset]->Update();
    }

  // Set the conjugate gradient filter to reconstruct from the first subset
  m_CG->SetInputProjectionStack(m_SubSelectFilters[0]->GetOutput());
  m_CG->SetInputProjectionWeights(m_SubSelectFilters[1]->GetOutput());
  m_CG->SetInputVolumeSeries(this->GetInputVolumeSeries());
  m_CG->SetNumberOfIterations(m_CG_iterations);
  m_CG->SetGeometry(m_SubSelectFilters[0]->GetOutputGeometry());
  m_CG->SetWeights(m_PhasesFilters[0]->GetOutput());
  m_CG->SetCudaConjugateGradient(this->GetCudaConjugateGradient());

  // Compute output information
  m_CG->UpdateOutputInformation();
  this->GetOutput()->CopyInformation( m_CG->GetOutput() );
}

template<class VolumeSeriesType, class ProjectionStackType>
void
IncrementalFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateData()
{
  typename VolumeSeriesType::Pointer pimg;

  for (unsigned int ml_iter=0; ml_iter < m_MainLoop_iterations; ml_iter++)
    {
    for (unsigned int subset = 0; subset < m_NumberOfSubsets; subset+=2)
      {
      // After the first update, we need to use the output as input
      if (subset + ml_iter > 0)
        {
        pimg = m_CG->GetOutput();
        pimg->DisconnectPipeline();
        m_CG->SetInputVolumeSeries(pimg);
        }

      // Change to the next subset
      m_CG->SetInputProjectionStack( m_SubSelectFilters[subset]->GetOutput() );
      m_CG->SetInputProjectionWeights( m_SubSelectFilters[subset+1]->GetOutput() );
      m_CG->SetGeometry( m_SubSelectFilters[subset]->GetOutputGeometry() );
      m_CG->SetWeights(m_PhasesFilters[subset]->GetOutput());
      TRY_AND_EXIT_ON_ITK_EXCEPTION( m_CG->Update() );
      }
    }


  this->GraftOutput( m_CG->GetOutput() );
}

} // end namespace rtk

#endif // __rtkIncrementalFourDConjugateGradientConeBeamReconstructionFilter_hxx
