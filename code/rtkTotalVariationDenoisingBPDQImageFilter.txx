#ifndef _rtkTotalVariationDenoisingBPDQImageFilter_txx
#define _rtkTotalVariationDenoisingBPDQImageFilter_txx
#include "rtkTotalVariationDenoisingBPDQImageFilter.h"

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
TotalVariationDenoisingBPDQImageFilter<TInputImage>
::TotalVariationDenoisingBPDQImageFilter()
{
    m_Lambda = 1.0;
    m_NumberOfIterations = 1;

    // Default behaviour is to process all dimensions
    this->m_DimensionsProcessed = new bool[TInputImage::ImageDimension];
    for (int dim = 0; dim < TInputImage::ImageDimension; dim++)
      {
        m_DimensionsProcessed[dim] = true;
      }

    // Create the sub filters
    m_GradientFilter = GradientFilterType::New();
    m_MultiplyFilter = MultiplyFilterType::New();
    m_SubtractImageFilter = SubtractImageFilterType::New();
    m_SubtractGradientFilter = SubtractGradientFilterType::New();
    m_MagnitudeThresholdFilter = MagnitudeThresholdFilterType::New();
    m_DivergenceFilter = DivergenceFilterType::New();

    // Set their initial connections
    // These connections change after the first iteration
    m_MultiplyFilter->SetInput1(m_GradientFilter->GetOutput());
    m_MagnitudeThresholdFilter->SetInput(m_MultiplyFilter->GetOutput());

    // Set whether the sub filters should release their data during pipeline execution
    m_DivergenceFilter->ReleaseDataFlagOn();
    // The m_SubtractImageFilter's output is the pipeline's output
    // Therefore it MUST NOT release its output
    m_SubtractImageFilter->ReleaseDataFlagOff();
    m_GradientFilter->ReleaseDataFlagOn();
    m_MultiplyFilter->ReleaseDataFlagOn();
    m_SubtractGradientFilter->ReleaseDataFlagOn();
    m_MagnitudeThresholdFilter->ReleaseDataFlagOn();
}

template< class TInputImage>
void
TotalVariationDenoisingBPDQImageFilter< TInputImage>
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
TotalVariationDenoisingBPDQImageFilter< TInputImage>
::GenerateOutputInformation()
{
  // get pointers to the input and output
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();

  // Set the inputs
  m_GradientFilter->SetInput(inputPtr);

  // Compute the parameters used in Basis Pursuit Dequantization
  // and set the filters to use them
  m_beta = 1;
  for (int dim=0; dim<TInputImage::ImageDimension; dim++)
    {
      if (m_DimensionsProcessed[dim])  m_beta /= 2;
    }
  m_beta -= 0.001; // Beta must be smaller than 2^(-NumberOfDimensionsProcessed) for the algorithm to converge

  m_gamma = 1 / (2 * m_Lambda); // BPDQ uses a cost function defined as 0.5 * || f - f_0 ||_2^2 + gamma * TV(f)
  m_MultiplyFilter->SetConstant2(-m_beta);
  m_MagnitudeThresholdFilter->SetThreshold(m_gamma);
  m_GradientFilter->SetDimensionsProcessed(m_DimensionsProcessed);
  m_DivergenceFilter->SetDimensionsProcessed(m_DimensionsProcessed);

  // Have the last filter calculate its output information,
  // which should update that of the whole pipeline
  m_MagnitudeThresholdFilter->UpdateOutputInformation();
  outputPtr->CopyInformation( m_MagnitudeThresholdFilter->GetOutput() );

  // Print the outputs of the filters

}


template< class TInputImage>
void
TotalVariationDenoisingBPDQImageFilter< TInputImage>
::GenerateData()
{
  // The first iteration only updates intermediate variables, not the output
  // The output is updated m_NumberOfIterations-1 times, therefore an additional
  // iteration must be performed so that even m_NumberOfIterations=1 has an effect
  for (int iter=0; iter<=m_NumberOfIterations; iter++)
    {
    m_MagnitudeThresholdFilter->Update();
    typename GradientImageType::Pointer pimg = m_MagnitudeThresholdFilter->GetOutput();
    pimg->DisconnectPipeline();
    m_DivergenceFilter->SetInput( pimg );
    m_SubtractGradientFilter->SetInput1( pimg );

    // Some subfilters are optimized out of the pipeline at the first iteration
    // After the first iteration, things go back to normal
    if(iter==0)
      {
      m_SubtractImageFilter->SetInput1(this->GetInput());
      m_SubtractImageFilter->SetInput2(m_DivergenceFilter->GetOutput());
      m_GradientFilter->SetInput(m_SubtractImageFilter->GetOutput());
      m_MultiplyFilter->SetConstant2(m_beta);
      m_SubtractGradientFilter->SetInput2(m_MultiplyFilter->GetOutput());
      m_MagnitudeThresholdFilter->SetInput(m_SubtractGradientFilter->GetOutput());
      }
    }

  this->GraftOutput(m_SubtractImageFilter->GetOutput());
}

} // end namespace rtk

#endif
