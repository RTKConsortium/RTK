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

#ifndef __rtkUnwarpSequenceImageFilter_hxx
#define __rtkUnwarpSequenceImageFilter_hxx

#include "rtkUnwarpSequenceImageFilter.h"

namespace rtk
{

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
UnwarpSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::UnwarpSequenceImageFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set the default values of member parameters
  m_NumberOfIterations=2;
  m_PhaseShift = 0;
  m_UseNearestNeighborInterpolationInWarping = false;
  m_CudaConjugateGradient = false;

  // Create the filters
  m_ZeroMultiplySequenceFilter = MultiplySequenceFilterType::New();
  m_ConjugateGradientFilter = ConjugateGradientFilterType::New();
#ifdef RTK_USE_CUDA
  if (m_CudaConjugateGradient)
    {
    m_ConjugateGradientFilter = rtk::CudaConjugateGradientImageFilter_4f::New();
    }
#endif
  m_WarpForwardFilter = WarpForwardFilterType::New();
  m_CGOperator = CGOperatorFilterType::New();

  // Set permanent connections
  m_ConjugateGradientFilter->SetB(m_WarpForwardFilter->GetOutput());
  m_ConjugateGradientFilter->SetX(m_ZeroMultiplySequenceFilter->GetOutput());
  m_ConjugateGradientFilter->SetA(m_CGOperator.GetPointer());

  // Set permanent parameters
  m_ZeroMultiplySequenceFilter->SetConstant2(itk::NumericTraits<typename TImageSequence::PixelType>::ZeroValue());
  m_WarpForwardFilter->SetForwardWarp(true);

  // Set memory management parameters
  m_ZeroMultiplySequenceFilter->ReleaseDataFlagOn();
  m_WarpForwardFilter->ReleaseDataFlagOn();
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
void
UnwarpSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::SetDisplacementField(const TMVFImageSequence* MVFs)
{
  this->SetNthInput(1, const_cast<TMVFImageSequence*>(MVFs));
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
typename TMVFImageSequence::Pointer
UnwarpSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::GetDisplacementField()
{
  return static_cast< TMVFImageSequence * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
void
UnwarpSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::GenerateInputRequestedRegion()
{
  //Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  //Get pointers to the input and output
  typename TImageSequence::Pointer  inputPtr  = const_cast<TImageSequence *>(this->GetInput(0));
  inputPtr->SetRequestedRegionToLargestPossibleRegion();

  typename TMVFImageSequence::Pointer  inputMVFPtr  = this->GetDisplacementField();
  inputMVFPtr->SetRequestedRegionToLargestPossibleRegion();
}


template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
void
UnwarpSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::GenerateOutputInformation()
{
  // Set runtime connections
  m_ZeroMultiplySequenceFilter->SetInput1(this->GetInput(0));
  m_CGOperator->SetDisplacementField(this->GetDisplacementField());
  m_CGOperator->SetUseNearestNeighborInterpolationInWarping(m_UseNearestNeighborInterpolationInWarping);
  m_WarpForwardFilter->SetInput(this->GetInput(0));
  m_WarpForwardFilter->SetDisplacementField(this->GetDisplacementField());
  m_WarpForwardFilter->SetUseNearestNeighborInterpolationInWarping(m_UseNearestNeighborInterpolationInWarping);

  // Set runtime parameters
  m_ConjugateGradientFilter->SetNumberOfIterations(this->m_NumberOfIterations);
  m_WarpForwardFilter->SetPhaseShift(this->m_PhaseShift);
  m_CGOperator->SetPhaseShift(this->m_PhaseShift);

  // Have the last filter calculate its output information
  m_ConjugateGradientFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_ConjugateGradientFilter->GetOutput() );
}

template< typename TImageSequence, typename TMVFImageSequence, typename TImage, typename TMVFImage>
void
UnwarpSequenceImageFilter< TImageSequence, TMVFImageSequence, TImage, TMVFImage>
::GenerateData()
{
  m_ConjugateGradientFilter->Update();

  this->GraftOutput( m_ConjugateGradientFilter->GetOutput() );
}

}// end namespace


#endif
