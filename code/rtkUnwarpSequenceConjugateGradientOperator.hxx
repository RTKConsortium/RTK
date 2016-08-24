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

#ifndef rtkUnwarpSequenceConjugateGradientOperator_hxx
#define rtkUnwarpSequenceConjugateGradientOperator_hxx

#include "rtkUnwarpSequenceConjugateGradientOperator.h"

namespace rtk
{

template< typename TImageSequence, typename TDVFImageSequence, typename TImage, typename TDVFImage>
UnwarpSequenceConjugateGradientOperator< TImageSequence, TDVFImageSequence, TImage, TDVFImage>
::UnwarpSequenceConjugateGradientOperator()
{
  this->SetNumberOfRequiredInputs(2);

  // Default member variables
  m_PhaseShift = 0;
  m_UseNearestNeighborInterpolationInWarping = false;
  m_UseCudaCyclicDeformation = false;

  // Create filters
  m_WarpSequenceBackwardFilter = WarpSequenceFilterType::New();
  m_WarpSequenceForwardFilter = WarpSequenceFilterType::New();

  // Set permanent connections
  m_WarpSequenceForwardFilter->SetInput(m_WarpSequenceBackwardFilter->GetOutput());

  // Set permanent parameters
  m_WarpSequenceForwardFilter->SetForwardWarp(true);

  // Set memory management options
  m_WarpSequenceBackwardFilter->ReleaseDataFlagOn();
}

template< typename TImageSequence, typename TDVFImageSequence, typename TImage, typename TDVFImage>
void
UnwarpSequenceConjugateGradientOperator< TImageSequence, TDVFImageSequence, TImage, TDVFImage>
::SetDisplacementField(const TDVFImageSequence* DVFs)
{
  this->SetNthInput(1, const_cast<TDVFImageSequence*>(DVFs));
}

template< typename TImageSequence, typename TDVFImageSequence, typename TImage, typename TDVFImage>
typename TDVFImageSequence::Pointer
UnwarpSequenceConjugateGradientOperator< TImageSequence, TDVFImageSequence, TImage, TDVFImage>
::GetDisplacementField()
{
  return static_cast< TDVFImageSequence * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template< typename TImageSequence, typename TDVFImageSequence, typename TImage, typename TDVFImage>
void
UnwarpSequenceConjugateGradientOperator< TImageSequence, TDVFImageSequence, TImage, TDVFImage>
::GenerateInputRequestedRegion()
{
  //Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  //Get pointers to the input and output
  typename TImageSequence::Pointer  inputPtr  = const_cast<TImageSequence *>(this->GetInput(0));
  inputPtr->SetRequestedRegionToLargestPossibleRegion();

  typename TDVFImageSequence::Pointer  inputDVFPtr  = this->GetDisplacementField();
  inputDVFPtr->SetRequestedRegionToLargestPossibleRegion();
}

template< typename TImageSequence, typename TDVFImageSequence, typename TImage, typename TDVFImage>
void
UnwarpSequenceConjugateGradientOperator< TImageSequence, TDVFImageSequence, TImage, TDVFImage>
::GenerateOutputInformation()
{
  // Set runtime connections, and connections with
  // forward and back projection filters, which are set
  // at runtime
  m_WarpSequenceBackwardFilter->SetInput(this->GetInput(0));
  m_WarpSequenceBackwardFilter->SetDisplacementField(this->GetDisplacementField());
  m_WarpSequenceForwardFilter->SetDisplacementField(this->GetDisplacementField());

  // Set runtime parameters
  m_WarpSequenceBackwardFilter->SetPhaseShift(this->m_PhaseShift);
  m_WarpSequenceBackwardFilter->SetUseNearestNeighborInterpolationInWarping(m_UseNearestNeighborInterpolationInWarping);
  m_WarpSequenceBackwardFilter->SetUseCudaCyclicDeformation(m_UseCudaCyclicDeformation);
  m_WarpSequenceForwardFilter->SetPhaseShift(this->m_PhaseShift);
  m_WarpSequenceForwardFilter->SetUseNearestNeighborInterpolationInWarping(m_UseNearestNeighborInterpolationInWarping);
  m_WarpSequenceForwardFilter->SetUseCudaCyclicDeformation(m_UseCudaCyclicDeformation);

  // Have the last filter calculate its output information
  m_WarpSequenceForwardFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_WarpSequenceForwardFilter->GetOutput() );
}

template< typename TImageSequence, typename TDVFImageSequence, typename TImage, typename TDVFImage>
void 
UnwarpSequenceConjugateGradientOperator< TImageSequence, TDVFImageSequence, TImage, TDVFImage>
::GenerateData()
{
  // Execute Pipeline
  m_WarpSequenceForwardFilter->Update();

  // Get the output
  this->GraftOutput( m_WarpSequenceForwardFilter->GetOutput() );
}

}// end namespace


#endif
