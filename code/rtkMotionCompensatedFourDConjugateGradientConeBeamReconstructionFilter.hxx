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

#ifndef __rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter_hxx
#define __rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter_hxx

#include "rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter.h"

namespace rtk
{

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(3);
  this->m_ProjStackToFourDFilter = MCProjStackToFourDType::New();
  this->m_CGOperator = MCCGOperatorType::New();
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::SetDisplacementField(const TMVFImageSequence* DisplacementField)
{
  this->SetNthInput(2, const_cast<TMVFImageSequence*>(DisplacementField));
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::SetInverseDisplacementField(const TMVFImageSequence* InverseDisplacementField)
{
  this->SetNthInput(3, const_cast<TMVFImageSequence*>(InverseDisplacementField));
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
typename TMVFImageSequence::ConstPointer
MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GetDisplacementField()
{
  return static_cast< const TMVFImageSequence * >
          ( this->itk::ProcessObject::GetInput(2) );
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
typename TMVFImageSequence::ConstPointer
MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GetInverseDisplacementField()
{
  return static_cast< const TMVFImageSequence * >
          ( this->itk::ProcessObject::GetInput(3) );
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
typename rtk::MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>::MCProjStackToFourDType*
MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GetProjectionStackToFourDFilter()
{
return(this->m_ProjStackToFourDFilter.GetPointer());
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
typename rtk::MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>::MCCGOperatorType*
MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GetConjugateGradientOperator()
{
return(this->m_CGOperator.GetPointer());
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::SetSignalFilename(const std::string _arg)
{
  GetProjectionStackToFourDFilter()->SetSignalFilename(_arg);
  GetConjugateGradientOperator()->SetSignalFilename(_arg);
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GenerateOutputInformation()
{
#ifdef RTK_USE_CUDA
  GetConjugateGradientOperator()->SetDisplacementField(this->GetDisplacementField());
  GetConjugateGradientOperator()->SetInverseDisplacementField(this->GetInverseDisplacementField());
  GetProjectionStackToFourDFilter()->SetDisplacementField(this->GetDisplacementField());
#endif

  Superclass::GenerateOutputInformation();
}

} // end namespace rtk

#endif // __rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter_hxx
