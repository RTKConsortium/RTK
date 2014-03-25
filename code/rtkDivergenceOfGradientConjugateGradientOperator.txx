#ifndef _rtkDivergenceOfGradientConjugateGradientOperator_txx
#define _rtkDivergenceOfGradientConjugateGradientOperator_txx
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
    this->m_DimensionsProcessed = new bool[TInputImage::ImageDimension];
    for (int dim = 0; dim < TInputImage::ImageDimension; dim++)
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
