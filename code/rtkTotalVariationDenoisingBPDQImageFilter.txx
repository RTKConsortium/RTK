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

#ifndef __rtkTotalVariationDenoisingBPDQImageFilter_txx
#define __rtkTotalVariationDenoisingBPDQImageFilter_txx

#include "rtkTotalVariationDenoisingBPDQImageFilter.h"

namespace rtk
{

template< typename TOutputImage, typename TGradientOutputImage> 
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientOutputImage>
::TotalVariationDenoisingBPDQImageFilter()
{
  m_Gamma = 1.0;
  m_NumberOfIterations = 1;

  // Default behaviour is to process all dimensions
  for (int dim = 0; dim < TOutputImage::ImageDimension; dim++)
    {
    m_DimensionsProcessed[dim] = true;
    }

  // Create the sub filters
  m_ZeroGradientFilter = GradientFilterType::New();
  m_GradientFilter = GradientFilterType::New();
  m_ZeroMultiplyFilter = MultiplyFilterType::New();
  m_MultiplyFilter = MultiplyFilterType::New();
  m_SubtractFilter = SubtractImageFilterType::New();
  m_SubtractGradientFilter = SubtractGradientFilterType::New();
  m_MagnitudeThresholdFilter = MagnitudeThresholdFilterType::New();
  m_DivergenceFilter = DivergenceFilterType::New();

  // Set permanent connections
  m_ZeroGradientFilter->SetInput(m_ZeroMultiplyFilter->GetOutput());
  m_SubtractFilter->SetInput2(m_DivergenceFilter->GetOutput());
  m_MultiplyFilter->SetInput1(m_SubtractFilter->GetOutput());
  m_GradientFilter->SetInput(m_MultiplyFilter->GetOutput());
  m_SubtractGradientFilter->SetInput2(m_GradientFilter->GetOutput());
  m_MagnitudeThresholdFilter->SetInput(m_SubtractGradientFilter->GetOutput());

  // Set permanent parameters
  m_ZeroMultiplyFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_BoundaryConditionForGradientFilter = new itk::ZeroFluxNeumannBoundaryCondition<TOutputImage>();
  m_BoundaryConditionForDivergenceFilter = new itk::ZeroFluxNeumannBoundaryCondition<TGradientOutputImage>();

  // Set whether the sub filters should release their data during pipeline execution
  m_ZeroMultiplyFilter->ReleaseDataFlagOn();
  m_ZeroGradientFilter->ReleaseDataFlagOff(); //Output used twice, but quick to compute and uses much memory
  m_DivergenceFilter->ReleaseDataFlagOn();
  // The m_SubtractFilter's output is the pipeline's output
  // Therefore it MUST NOT release its output
  m_SubtractFilter->ReleaseDataFlagOff();
  m_GradientFilter->ReleaseDataFlagOn();
  m_MultiplyFilter->ReleaseDataFlagOn();
  m_SubtractGradientFilter->ReleaseDataFlagOn();
  m_MagnitudeThresholdFilter->ReleaseDataFlagOn();
}

template< typename TOutputImage, typename TGradientOutputImage>
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientOutputImage>
::~TotalVariationDenoisingBPDQImageFilter()
{
  delete m_BoundaryConditionForGradientFilter;
  delete m_BoundaryConditionForDivergenceFilter;
}


template< typename TOutputImage, typename TGradientOutputImage> 
void
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientOutputImage>
::SetDimensionsProcessed(bool* arg)
{
  bool Modified=false;
  for (int dim=0; dim<TOutputImage::ImageDimension; dim++)
    {
    if (m_DimensionsProcessed[dim] != arg[dim])
      {
      m_DimensionsProcessed[dim] = arg[dim];
      Modified = true;
      }
    }
  if(Modified) this->Modified();
}

template< typename TOutputImage, typename TGradientOutputImage>
void
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientOutputImage>
::SetBoundaryConditionToPeriodic()
{
  m_BoundaryConditionForGradientFilter = new itk::PeriodicBoundaryCondition<TOutputImage>();
  m_BoundaryConditionForDivergenceFilter = new itk::PeriodicBoundaryCondition<TGradientOutputImage>();

  m_GradientFilter->OverrideBoundaryCondition(m_BoundaryConditionForGradientFilter);
  m_DivergenceFilter->OverrideBoundaryCondition(m_BoundaryConditionForDivergenceFilter);
}

template< typename TOutputImage, typename TGradientOutputImage> 
void
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientOutputImage>
::GenerateOutputInformation()
{
  // get pointers to the input and output
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();

  // Set the inputs
  m_ZeroMultiplyFilter->SetInput(inputPtr);
  m_SubtractFilter->SetInput1(inputPtr);

  // These connections must be made at runtime, because if
  // the filter is updated, then re-used with a different input,
  // these connections must be reset
  m_DivergenceFilter->SetInput(m_ZeroGradientFilter->GetOutput());
  m_SubtractGradientFilter->SetInput1(m_ZeroGradientFilter->GetOutput());

  // Compute the parameters used in Basis Pursuit Dequantization
  // and set the filters to use them
  double numberOfDimensionsProcessed = 0;
  for (int dim=0; dim<TOutputImage::ImageDimension; dim++)
    {
    if (m_DimensionsProcessed[dim])  numberOfDimensionsProcessed += 1.0;
    }
  m_Beta = 1/(2 * numberOfDimensionsProcessed) - 0.001; // Beta must be smaller than 1 / (2 * NumberOfDimensionsProcessed) for the algorithm to converge

  m_MultiplyFilter->SetConstant2(m_Beta);
  m_MagnitudeThresholdFilter->SetThreshold(m_Gamma);
  m_GradientFilter->SetDimensionsProcessed(m_DimensionsProcessed);
  m_ZeroGradientFilter->SetDimensionsProcessed(m_DimensionsProcessed);
  m_DivergenceFilter->SetDimensionsProcessed(m_DimensionsProcessed);

  // Have the last filter calculate its output information,
  // which should update that of the whole pipeline
  m_MagnitudeThresholdFilter->UpdateOutputInformation();
  outputPtr->CopyInformation( m_SubtractFilter->GetOutput() );
}


template< typename TOutputImage, typename TGradientOutputImage> 
void
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientOutputImage>
::GenerateData()
{
  typename TGradientOutputImage::Pointer pimg;

  // The first iteration only updates intermediate variables, not the output
  // The output is updated m_NumberOfIterations-1 times, therefore an additional
  // iteration must be performed so that even m_NumberOfIterations=1 has an effect
  for (int iter=0; iter<=m_NumberOfIterations; iter++)
    {
    m_MagnitudeThresholdFilter->Update();
    pimg = m_MagnitudeThresholdFilter->GetOutput();
    pimg->DisconnectPipeline();
    m_DivergenceFilter->SetInput( pimg );
    m_SubtractGradientFilter->SetInput1( pimg );
    }
  this->GraftOutput(m_SubtractFilter->GetOutput());

  //Release the data in pimg
  pimg->ReleaseData();
}

} // end namespace rtk

#endif
