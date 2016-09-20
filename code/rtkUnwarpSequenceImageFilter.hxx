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

#ifndef rtkUnwarpSequenceImageFilter_hxx
#define rtkUnwarpSequenceImageFilter_hxx

#include "rtkUnwarpSequenceImageFilter.h"

namespace rtk
{

template< typename TImageSequence, typename TDVFImageSequence, typename TImage, typename TDVFImage>
UnwarpSequenceImageFilter< TImageSequence, TDVFImageSequence, TImage, TDVFImage>
::UnwarpSequenceImageFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set the default values of member parameters
  m_NumberOfIterations=2;
  m_PhaseShift = 0;
  m_UseNearestNeighborInterpolationInWarping = false;
  m_CudaConjugateGradient = false;
  m_UseCudaCyclicDeformation = false;

  // Create the filters
  m_ConjugateGradientFilter = ConjugateGradientFilterType::New();
  m_ConstantSource = ConstantSourceType::New();
#ifdef RTK_USE_CUDA
  if (m_CudaConjugateGradient)
    {
    m_ConjugateGradientFilter = rtk::CudaConjugateGradientImageFilter_4f::New();
    }
    m_ConstantSource = rtk::CudaConstantVolumeSeriesSource::New();
#endif
  m_WarpForwardFilter = WarpForwardFilterType::New();
  m_CGOperator = CGOperatorFilterType::New();
  
  // Set permanent connections
  m_ConjugateGradientFilter->SetB(m_WarpForwardFilter->GetOutput());
  m_ConjugateGradientFilter->SetX(m_ConstantSource->GetOutput());
  m_ConjugateGradientFilter->SetA(m_CGOperator.GetPointer());

  // Set permanent parameters
  m_WarpForwardFilter->SetForwardWarp(true);

  // Set memory management parameters
  m_ConstantSource->ReleaseDataFlagOn();
  m_WarpForwardFilter->ReleaseDataFlagOn();
}

template< typename TImageSequence, typename TDVFImageSequence, typename TImage, typename TDVFImage>
void
UnwarpSequenceImageFilter< TImageSequence, TDVFImageSequence, TImage, TDVFImage>
::SetDisplacementField(const TDVFImageSequence* DVFs)
{
  this->SetNthInput(1, const_cast<TDVFImageSequence*>(DVFs));
}

template< typename TImageSequence, typename TDVFImageSequence, typename TImage, typename TDVFImage>
typename TDVFImageSequence::Pointer
UnwarpSequenceImageFilter< TImageSequence, TDVFImageSequence, TImage, TDVFImage>
::GetDisplacementField()
{
  return static_cast< TDVFImageSequence * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template< typename TImageSequence, typename TDVFImageSequence, typename TImage, typename TDVFImage>
void
UnwarpSequenceImageFilter< TImageSequence, TDVFImageSequence, TImage, TDVFImage>
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
UnwarpSequenceImageFilter< TImageSequence, TDVFImageSequence, TImage, TDVFImage>
::GenerateOutputInformation()
{
  // Set runtime connections
  m_ConstantSource->SetInformationFromImage(this->GetInput(0));
  m_CGOperator->SetDisplacementField(this->GetDisplacementField());
  m_CGOperator->SetUseNearestNeighborInterpolationInWarping(m_UseNearestNeighborInterpolationInWarping);
  m_WarpForwardFilter->SetInput(this->GetInput(0));
  m_WarpForwardFilter->SetDisplacementField(this->GetDisplacementField());
  m_WarpForwardFilter->SetUseNearestNeighborInterpolationInWarping(m_UseNearestNeighborInterpolationInWarping);
  m_WarpForwardFilter->SetUseCudaCyclicDeformation(m_UseCudaCyclicDeformation);

  // Set runtime parameters
  m_ConjugateGradientFilter->SetNumberOfIterations(this->m_NumberOfIterations);
  m_WarpForwardFilter->SetPhaseShift(this->m_PhaseShift);
  m_CGOperator->SetPhaseShift(this->m_PhaseShift);
  m_CGOperator->SetUseCudaCyclicDeformation(m_UseCudaCyclicDeformation);

  // Have the last filter calculate its output information
  m_ConjugateGradientFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_ConjugateGradientFilter->GetOutput() );
}

template< typename TImageSequence, typename TDVFImageSequence, typename TImage, typename TDVFImage>
void
UnwarpSequenceImageFilter< TImageSequence, TDVFImageSequence, TImage, TDVFImage>
::GenerateData()
{
  m_ConjugateGradientFilter->Update();

  this->GraftOutput( m_ConjugateGradientFilter->GetOutput() );
  
  // During mini-pipeline execution, the requested region on the primary input 
  // is modified by the extract filters contained in the warp filters. This 
  typename TImageSequence::Pointer  inputPtr  = const_cast<TImageSequence *>(this->GetInput(0));
  inputPtr->SetRequestedRegionToLargestPossibleRegion();

  typename TDVFImageSequence::Pointer  inputDVFPtr  = this->GetDisplacementField();
  inputDVFPtr->SetRequestedRegionToLargestPossibleRegion();
}

}// end namespace


#endif
