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

#ifndef rtkDivergenceOfGradientConjugateGradientOperator_hxx
#define rtkDivergenceOfGradientConjugateGradientOperator_hxx
#include "rtkDivergenceOfGradientConjugateGradientOperator.h"

#include "itkConstShapedNeighborhoodIterator.h"
#include "itkNeighborhoodInnerProduct.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkNeighborhoodAlgorithm.h"
#include "itkPeriodicBoundaryCondition.h"
#include "itkOffset.h"
#include "itkProgressReporter.h"

namespace rtk
{

template <class TInputImage>
DivergenceOfGradientConjugateGradientOperator<TInputImage>
::DivergenceOfGradientConjugateGradientOperator()
{
  // Default behaviour is to process all dimensions
  for (unsigned int dim = 0; dim < TInputImage::ImageDimension; dim++)
    {
    m_DimensionsProcessed[dim] = true;
    }

  // Create the sub filters
  m_GradientFilter = GradientFilterType::New();
  m_DivergenceFilter = DivergenceFilterType::New();

  // Set their initial connections
  // These connections change after the first iteration
  m_DivergenceFilter->SetInput(m_GradientFilter->GetOutput());

  // Set whether the sub filters should release their data during pipeline execution
  m_GradientFilter->ReleaseDataFlagOn();
}

template< class TInputImage>
void
DivergenceOfGradientConjugateGradientOperator< TInputImage>
::SetDimensionsProcessed(bool* arg){
  bool Modified=false;
  for (int dim=0; dim<TInputImage::ImageDimension; dim++)
    {
    if (m_DimensionsProcessed[dim] != arg[dim])
      {
      m_DimensionsProcessed[dim] = arg[dim];
      Modified = true;
      }
    }
  if(Modified) this->Modified();
}

template< class TInputImage>
void
DivergenceOfGradientConjugateGradientOperator< TInputImage>
::GenerateOutputInformation()
{
  // get pointers to the input and output
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();

  // Set the inputs
  m_GradientFilter->SetInput(inputPtr);

  m_GradientFilter->SetDimensionsProcessed(m_DimensionsProcessed);
  m_DivergenceFilter->SetDimensionsProcessed(m_DimensionsProcessed);

  // Have the last filter calculate its output information,
  m_DivergenceFilter->UpdateOutputInformation();
  outputPtr->CopyInformation( m_DivergenceFilter->GetOutput() );
}


template< class TInputImage>
void
DivergenceOfGradientConjugateGradientOperator< TInputImage>
::GenerateData()
{
  m_DivergenceFilter->Update();
  this->GraftOutput(m_DivergenceFilter->GetOutput());
}

} // end namespace rtk

#endif
