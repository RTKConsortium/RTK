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

#ifndef __rtkReconstructionConjugateGradientOperator_txx
#define __rtkReconstructionConjugateGradientOperator_txx

#include "rtkReconstructionConjugateGradientOperator.h"

namespace rtk
{

template< typename TOutputImage>
ReconstructionConjugateGradientOperator<TOutputImage>::ReconstructionConjugateGradientOperator()
{
  this->SetNumberOfRequiredInputs(2);
  m_MultiplyFilter = MultiplyFilterType::New();
  m_ZeroMultiplyProjectionFilter = MultiplyFilterType::New();
  m_ZeroMultiplyVolumeFilter = MultiplyFilterType::New();

  // Set permanent parameters
  m_ZeroMultiplyProjectionFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_ZeroMultiplyVolumeFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());

  // Set memory management options
  m_ZeroMultiplyProjectionFilter->ReleaseDataFlagOn();
  m_ZeroMultiplyVolumeFilter->ReleaseDataFlagOn();
  m_MultiplyFilter->ReleaseDataFlagOn();
}

template< typename TOutputImage >
void
ReconstructionConjugateGradientOperator<TOutputImage>
::SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg)
{
  m_BackProjectionFilter = _arg;
}

template< typename TOutputImage >
void
ReconstructionConjugateGradientOperator<TOutputImage>
::SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg)
{
  m_ForwardProjectionFilter = _arg;
}


template< typename TOutputImage >
void
ReconstructionConjugateGradientOperator<TOutputImage>
::SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg)
{
  m_BackProjectionFilter->SetGeometry(_arg.GetPointer());
  m_ForwardProjectionFilter->SetGeometry(_arg);
}

template< typename TOutputImage >
void
ReconstructionConjugateGradientOperator<TOutputImage>
::GenerateInputRequestedRegion()
{
  // Input 0 is the volume in which we backproject
  typename Superclass::InputImagePointer inputPtr0 =
    const_cast< TOutputImage * >( this->GetInput(0) );
  if ( !inputPtr0 ) return;
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the stack of projections to backproject
  typename Superclass::InputImagePointer  inputPtr1 =
    const_cast< TOutputImage * >( this->GetInput(1) );
  if ( !inputPtr1 ) return;
  inputPtr1->SetRequestedRegion( inputPtr1->GetLargestPossibleRegion() );
}

template< typename TOutputImage >
void
ReconstructionConjugateGradientOperator<TOutputImage>
::GenerateOutputInformation()
{
  // Set runtime connections, and connections with
  // forward and back projection filters, which are set
  // at runtime
  m_ForwardProjectionFilter->SetInput(0, m_ZeroMultiplyProjectionFilter->GetOutput());
  m_BackProjectionFilter->SetInput(0, m_ZeroMultiplyVolumeFilter->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_ForwardProjectionFilter->GetOutput());
  m_ZeroMultiplyVolumeFilter->SetInput1(this->GetInput(0));
  m_ZeroMultiplyProjectionFilter->SetInput1(this->GetInput(1));
  m_ForwardProjectionFilter->SetInput(1, this->GetInput(0));

  // Set memory management parameters for forward
  // and back projection filters
  m_ForwardProjectionFilter->ReleaseDataFlagOn();
  m_BackProjectionFilter->ReleaseDataFlagOn();

  // Have the last filter calculate its output information
  m_BackProjectionFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_BackProjectionFilter->GetOutput() );
}

template< typename TOutputImage >
void ReconstructionConjugateGradientOperator<TOutputImage>::GenerateData()
{
  // Execute Pipeline
  m_BackProjectionFilter->Update();

  // Get the output
  this->GraftOutput( m_BackProjectionFilter->GetOutput() );
}

}// end namespace


#endif
