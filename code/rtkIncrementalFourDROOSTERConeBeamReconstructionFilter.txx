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

#ifndef __rtkIncrementalFourDROOSTERConeBeamReconstructionFilter_txx
#define __rtkIncrementalFourDROOSTERConeBeamReconstructionFilter_txx

#include "rtkIncrementalFourDROOSTERConeBeamReconstructionFilter.h"
#include <algorithm>

namespace rtk
{

template<class VolumeSeriesType, class ProjectionStackType>
IncrementalFourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::IncrementalFourDROOSTERConeBeamReconstructionFilter()
{
  // Set the default values of member parameters
  m_NumberOfProjectionsPerSubset=0;
  m_NumberOfSubsets=1;
  m_Kzero = 1;
}

template<class VolumeSeriesType, class ProjectionStackType>
void
IncrementalFourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  // Call the superclass method to prepare all filters of the pipeline with
  // the correct runtime parameters, but without connecting them to each other
  this->PreparePipeline();

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

  for (unsigned int subset = 0; subset < m_NumberOfSubsets; subset++)
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
    m_SubSelectFilters[subset]->SetInputGeometry( this->m_Geometry );
    m_SubSelectFilters[subset]->SetInputProjectionStack( this->GetInputProjectionStack() );
    m_SubSelectFilters[subset]->Update();

    // Read the phases file, and extract the interpolation weights for this subset
    m_PhasesFilters.push_back(rtk::PhasesToInterpolationWeights::New());
    m_PhasesFilters[subset]->SetFileName(m_PhasesFileName);
    m_PhasesFilters[subset]->SetNumberOfReconstructedFrames(this->GetInputVolumeSeries()->GetLargestPossibleRegion().GetSize(Dimension));
    m_PhasesFilters[subset]->SetSelectedProjections( m_SelectedProjsVector[subset] );
    m_PhasesFilters[subset]->Update();
    }

  // Set the conjugate gradient filter to reconstruct from the first subset
  this->m_FourDCGFilter->SetInputProjectionStack(m_SubSelectFilters[0]->GetOutput());
  this->m_FourDCGFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());
  this->m_FourDCGFilter->SetNumberOfIterations(this->m_CG_iterations);
  this->m_FourDCGFilter->SetGeometry(m_SubSelectFilters[0]->GetOutputGeometry());
  this->m_FourDCGFilter->SetWeights(m_PhasesFilters[0]->GetOutput());

  // Compute output information
  this->m_FourDCGFilter->UpdateOutputInformation();
  this->GetOutput()->CopyInformation( this->m_FourDCGFilter->GetOutput() );
}

template<class VolumeSeriesType, class ProjectionStackType>
void
IncrementalFourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
::GenerateData()
{
  typename VolumeSeriesType::Pointer pimg;

  // Variables to loop through the data subterms, the regularizations and the constraints
  unsigned int constraintIndex = 0;

  // Real work
  for (unsigned int ml_iter=0; ml_iter < this->m_MainLoop_iterations; ml_iter++)
    {
    for (unsigned int costTerm = 0; costTerm < m_NumberOfSubsets + 2; costTerm++)
      {
      float alpha_k = m_Kzero / ((ml_iter * m_NumberOfSubsets + 2) + m_Kzero);

      if (costTerm < m_NumberOfSubsets)
        {
        // If this is the first subset during the first main loop iteration,
        // pimg does not contain anything yet. Use the filter's input instead
        if ((costTerm == 0) && (ml_iter == 0))
          this->m_FourDCGFilter->SetInputVolumeSeries( this->GetInputVolumeSeries() );
        else
          this->m_FourDCGFilter->SetInputVolumeSeries( pimg );

        this->m_FourDCGFilter->SetInputProjectionStack( m_SubSelectFilters[costTerm]->GetOutput() );
        this->m_FourDCGFilter->SetGeometry( m_SubSelectFilters[costTerm]->GetOutputGeometry() );
        this->m_FourDCGFilter->SetWeights(m_PhasesFilters[costTerm]->GetOutput());
        TRY_AND_EXIT_ON_ITK_EXCEPTION( this->m_FourDCGFilter->Update() );
        pimg = this->m_FourDCGFilter->GetOutput();
        pimg->DisconnectPipeline();
        }
      if (costTerm == m_NumberOfSubsets)
        {
        // Spatial TV
        this->m_TVDenoisingSpace->SetGamma(this->m_GammaSpace * alpha_k);
        this->m_TVDenoisingSpace->SetInput( pimg );
        TRY_AND_EXIT_ON_ITK_EXCEPTION( this->m_TVDenoisingSpace->Update() );
        pimg = this->m_TVDenoisingSpace->GetOutput();
        pimg->DisconnectPipeline();
        }
      if (costTerm == m_NumberOfSubsets + 1)
        {
        if (this->m_PerformWarping)
          {
          // Warp all frames to a single phase
          this->m_Warp->SetInput( pimg );
          TRY_AND_EXIT_ON_ITK_EXCEPTION( this->m_Warp->Update(); );
          pimg = this->m_Warp->GetOutput();
          pimg->DisconnectPipeline();
          }

        // Temporal TV
        this->m_TVDenoisingTime->SetGamma(this->m_GammaTime * alpha_k);
        this->m_TVDenoisingTime->SetInput( pimg );
        TRY_AND_EXIT_ON_ITK_EXCEPTION( this->m_TVDenoisingTime->Update() );
        pimg = this->m_TVDenoisingTime->GetOutput();
        pimg->DisconnectPipeline();

        if (this->m_PerformWarping)
          {
          // Warp all frames back
          this->m_Unwarp->SetInput( pimg );
          TRY_AND_EXIT_ON_ITK_EXCEPTION( this->m_Unwarp->Update(); );
          pimg = this->m_Unwarp->GetOutput();
          pimg->DisconnectPipeline();
          }
        }

      // Apply one constraint
      if (constraintIndex%2)
        {
        this->m_AverageOutOfROIFilter->SetInput(pimg);
        TRY_AND_EXIT_ON_ITK_EXCEPTION( this->m_AverageOutOfROIFilter->Update() );
        pimg = this->m_AverageOutOfROIFilter->GetOutput();
        pimg->DisconnectPipeline();
        }
      else
        {
        this->m_PositivityFilter->SetInput(pimg);
        TRY_AND_EXIT_ON_ITK_EXCEPTION( this->m_PositivityFilter->Update() );
        pimg = this->m_PositivityFilter->GetOutput();
        pimg->DisconnectPipeline();
        }

      // Increment the constraint index
      constraintIndex++;
      }
    }

  this->GraftOutput( pimg );
}

} // end namespace rtk

#endif // __rtkIncrementalFourDROOSTERConeBeamReconstructionFilter_txx
