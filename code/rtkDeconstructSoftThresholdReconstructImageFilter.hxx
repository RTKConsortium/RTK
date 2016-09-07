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

#ifndef rtkDeconstructSoftThresholdReconstructImageFilter_hxx
#define rtkDeconstructSoftThresholdReconstructImageFilter_hxx

//rtk Includes
#include "rtkDeconstructSoftThresholdReconstructImageFilter.h"

namespace rtk
{

/////////////////////////////////////////////////////////
//Constructor()
template <class TImage>
DeconstructSoftThresholdReconstructImageFilter<TImage>
::DeconstructSoftThresholdReconstructImageFilter()
{
    m_DeconstructionFilter = DeconstructFilterType::New();
    m_ReconstructionFilter = ReconstructFilterType::New();
    m_Order = 3;
    m_Threshold = 0;
    m_PipelineConstructed = false;
}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void
DeconstructSoftThresholdReconstructImageFilter<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}

/////////////////////////////////////////////////////////
// Pass the number of decomposition levels to the wavelet
// filters
template <class TImage>
void
DeconstructSoftThresholdReconstructImageFilter<TImage>
::SetNumberOfLevels(unsigned int levels)
{
    m_DeconstructionFilter->SetNumberOfLevels(levels);
    m_ReconstructionFilter->SetNumberOfLevels(levels);
}

/////////////////////////////////////////////////////////
//GenerateInputRequestedRegion()
template <class TImage>
void
DeconstructSoftThresholdReconstructImageFilter<TImage>
::GenerateInputRequestedRegion()
{
  InputImagePointer  inputPtr  = const_cast<TImage *>(this->GetInput());
  inputPtr->SetRequestedRegionToLargestPossibleRegion();
}

/////////////////////////////////////////////////////////
//GenerateOutputInformation()
template <class TImage>
void
DeconstructSoftThresholdReconstructImageFilter<TImage>
::GenerateOutputInformation()
{

  if (!m_PipelineConstructed)
    {
    // Connect the inputs
    m_DeconstructionFilter->SetInput(this->GetInput());
    m_DeconstructionFilter->ReleaseDataFlagOn();

    // Set runtime parameters
    m_DeconstructionFilter->SetOrder(this->GetOrder());
    m_ReconstructionFilter->SetOrder(this->GetOrder());
    m_DeconstructionFilter->UpdateOutputInformation();
    m_ReconstructionFilter->SetSizes(m_DeconstructionFilter->GetSizes());
    m_ReconstructionFilter->SetIndices(m_DeconstructionFilter->GetIndices());

    //Create and setup an array of soft threshold filters
    for (unsigned int index=0; index < m_DeconstructionFilter->GetNumberOfOutputs(); index++)
      {
      // Soft thresholding
      m_SoftTresholdFilters.push_back(SoftThresholdFilterType::New());
      m_SoftTresholdFilters[index]->SetInput(m_DeconstructionFilter->GetOutput(index));
      m_SoftTresholdFilters[index]->SetThreshold(m_Threshold);
      m_SoftTresholdFilters[index]->ReleaseDataFlagOn();

      //Set input for reconstruction
      m_ReconstructionFilter->SetInput(index, m_SoftTresholdFilters[index]->GetOutput());
      }

    // The low pass coefficients are not thresholded
    m_SoftTresholdFilters[0]->SetThreshold(0);
    }

  m_PipelineConstructed = true;

  // Have the last filter calculate its output information
  // and copy it as the output information of the composite filter
  m_ReconstructionFilter->UpdateOutputInformation();
  this->GetOutput()->CopyInformation( m_ReconstructionFilter->GetOutput() );
}

/////////////////////////////////////////////////////////
//GenerateData()
template <class TImage>
void
DeconstructSoftThresholdReconstructImageFilter<TImage>
::GenerateData()
{
  // Perform reconstruction
  m_ReconstructionFilter->Update();
  this->GraftOutput( m_ReconstructionFilter->GetOutput() );
}


}// end namespace rtk

#endif
