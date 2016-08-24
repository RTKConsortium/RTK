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
#ifndef rtkMotionCompensatedFourDROOSTERConeBeamReconstructionFilter_hxx
#define rtkMotionCompensatedFourDROOSTERConeBeamReconstructionFilter_hxx

#include "rtkMotionCompensatedFourDROOSTERConeBeamReconstructionFilter.h"
#include <itkImageFileWriter.h>

namespace rtk
{

template<typename VolumeSeriesType, typename ProjectionStackType>
MotionCompensatedFourDROOSTERConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType>
::MotionCompensatedFourDROOSTERConeBeamReconstructionFilter()
{
  // Create the filters
#ifdef RTK_USE_CUDA
  this->m_FourDCGFilter = MotionCompensatedFourDCGFilterType::New();
#endif
}

template< typename VolumeSeriesType, typename ProjectionStackType>
void
MotionCompensatedFourDROOSTERConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType>
::SetSignal(const std::vector<double> signal)
{
  this->m_FourDCGFilter->SetSignal(signal);
}

template<typename VolumeSeriesType, typename ProjectionStackType>
void
MotionCompensatedFourDROOSTERConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType>
::GenerateInputRequestedRegion()
{
  // Set variables so that the superclass' implementation
  // generates the correct input requested regions
  this->m_PerformWarping = true;
  this->m_ComputeInverseWarpingByConjugateGradient = false;

  //Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();
}

template<typename VolumeSeriesType, typename ProjectionStackType>
void
MotionCompensatedFourDROOSTERConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType>
::GenerateOutputInformation()
{
  // Set m_PerformWarping to false so as not
  // to plug any of the warping filters into the pipeline
  // (DVFs are taken into account in the motion compensated 4D conjugate gradient filter)
  this->m_PerformWarping = false;

  // Set the 4D conjugate gradient filter
  dynamic_cast<MotionCompensatedFourDCGFilterType*>(this->m_FourDCGFilter.GetPointer())->SetDisplacementField(this->GetDisplacementField());
  dynamic_cast<MotionCompensatedFourDCGFilterType*>(this->m_FourDCGFilter.GetPointer())->SetInverseDisplacementField(this->GetInverseDisplacementField());
  dynamic_cast<MotionCompensatedFourDCGFilterType*>(this->m_FourDCGFilter.GetPointer())->SetUseCudaCyclicDeformation(this->m_UseCudaCyclicDeformation);

  // Call the superclass implementation
  Superclass::GenerateOutputInformation();
}

template<typename VolumeSeriesType, typename ProjectionStackType>
void
MotionCompensatedFourDROOSTERConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType>
::GenerateData()
{
  // Set the variables so that the superclass implementation updates the right filters
  this->m_PerformWarping = false;

  // Call the superclass implementation
  Superclass::GenerateData();
}

}// end namespace


#endif
