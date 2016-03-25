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

//#ifdef RTK_USE_CUDA
//  this->m_ForwardProjectionFilter = CudaWarpForwardProjectionImageFilter::New();
//  this->m_BackProjectionFilter = CudaWarpBackProjectionImageFilter::New();
//#else
//  this->SetForwardProjectionFilter(0);
//  this->SetBackProjectionFilter(0);
//#endif
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

//template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
//void
//MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
//::SetWeights(const itk::Array2D<float> _arg)
//{
//  m_ProjStackToFourDFilter->SetWeights(_arg);
//  m_CGOperator->SetWeights(_arg);
//  this->Modified();
//}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::SetSignalFilename(const std::string _arg)
{
  m_ProjStackToFourDFilter->SetSignalFilename(_arg);
  m_CGOperator->SetSignalFilename(_arg);
}



template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
void
MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
::GenerateOutputInformation()
{
#ifdef RTK_USE_CUDA
  m_CGOperator->SetDisplacementField(this->GetInverseDisplacementField());
  m_ProjStackToFourDFilter->SetDisplacementField(this->GetDisplacementField());
#endif

  Superclass::GenerateOutputInformation();
}

//template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
//void
//MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
//::GenerateInputRequestedRegion()
//{
//  //Call the superclass' implementation of this method
//  Superclass::GenerateInputRequestedRegion();

//  this->m_ProjStackToFourDFilter->PropagateRequestedRegion(this->m_ProjStackToFourDFilter->GetOutput());
//}

//template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
//void
//MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
//::GenerateData()
//{
//  m_ProjStackToFourDFilter->Update();

//  // If m_ProjStackToFourDFilter->GetOutput() is stored in an itk::CudaImage, make sure its data is transferred on the CPU
//  this->m_ProjStackToFourDFilter->GetOutput()->GetBufferPointer();

//  m_ConjugateGradientFilter->Update();

//  // Simply grafting the output of m_ConjugateGradientFilter to the main output
//  // is sufficient in most cases, but when this output is then disconnected and replugged,
//  // several images end up having the same CudaDataManager. The following solution is a
//  // workaround for this problem
//  typename VolumeSeriesType::Pointer pimg = m_ConjugateGradientFilter->GetOutput();
//  pimg->DisconnectPipeline();

//  this->GraftOutput( pimg);
//}

//template< typename VolumeSeriesType, typename ProjectionStackType, typename TMVFImageSequence, typename TMVFImage>
//void
//MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>
//::PrintTiming(std::ostream& os) const
//{
//}

} // end namespace rtk

#endif // __rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter_hxx
