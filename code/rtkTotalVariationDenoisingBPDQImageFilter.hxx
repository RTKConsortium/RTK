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

#ifndef __rtkTotalVariationDenoisingBPDQImageFilter_hxx
#define __rtkTotalVariationDenoisingBPDQImageFilter_hxx

#include "rtkTotalVariationDenoisingBPDQImageFilter.h"

namespace rtk
{

template< typename TOutputImage, typename TGradientImage>
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::TotalVariationDenoisingBPDQImageFilter()
{
  m_Gamma = 1.0;
  m_NumberOfIterations = 1;

  // This is an InPlace filter only for the subclasses to have the possibility to run in place
  this->SetInPlace(false);

  // Default behaviour is to process all dimensions
  for (int dim = 0; dim < TOutputImage::ImageDimension; dim++)
    {
    m_DimensionsProcessed[dim] = true;
    }

  // Create the sub filters
  m_GradientFilter = GradientFilterType::New();
  m_MultiplyFilter = MultiplyFilterType::New();
  m_SubtractFilter = SubtractImageFilterType::New();
  m_SubtractGradientFilter = SubtractGradientFilterType::New();
  m_MagnitudeThresholdFilter = MagnitudeThresholdFilterType::New();
  m_DivergenceFilter = DivergenceFilterType::New();

  // Set permanent parameters
  m_BoundaryConditionForGradientFilter = new itk::ZeroFluxNeumannBoundaryCondition<TOutputImage>();
  m_BoundaryConditionForDivergenceFilter = new itk::ZeroFluxNeumannBoundaryCondition<TGradientImage>();

  // Set whether the sub filters should release their data during pipeline execution
  m_DivergenceFilter->ReleaseDataFlagOn();
  m_SubtractFilter->ReleaseDataFlagOn(); // It is the pipeline's output, but it is explicitely computed during the last iteration
  m_GradientFilter->ReleaseDataFlagOn();
  m_MultiplyFilter->ReleaseDataFlagOn();
  m_SubtractGradientFilter->ReleaseDataFlagOn();

  // Set some filters to be InPlace

  // TotalVariationDenoisingBPDQ reaches its memory consumption peak
  // when m_SubtractGradientFilter allocates its output (a covariant vector image)
  // and uses its two inputs (two covariant vector images)
  // Setting it in place reduces the memory requirement from 3 covariant vector images to 2
  m_SubtractGradientFilter->SetInPlace(true);
}

template< typename TOutputImage, typename TGradientImage>
void
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientImage>
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

template< typename TOutputImage, typename TGradientImage>
void
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::SetBoundaryConditionToPeriodic()
{
  m_BoundaryConditionForGradientFilter = new itk::PeriodicBoundaryCondition<TOutputImage>();
  m_BoundaryConditionForDivergenceFilter = new itk::PeriodicBoundaryCondition<TGradientImage>();
  m_GradientFilter->OverrideBoundaryCondition(m_BoundaryConditionForGradientFilter);
  m_DivergenceFilter->OverrideBoundaryCondition(m_BoundaryConditionForDivergenceFilter);
}

template< typename TOutputImage, typename TGradientImage>
void
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::GenerateOutputInformation()
{
  // Set the pipeline for the first iteration
  SetPipelineForFirstIteration();

  // Compute the parameters used in Basis Pursuit Dequantization
  // and set the filters to use them
  double numberOfDimensionsProcessed = 0;
  float maxSpacing = 0;
  for (int dim=0; dim<TOutputImage::ImageDimension; dim++)
    {
    if (m_DimensionsProcessed[dim])
      {
      numberOfDimensionsProcessed += 1.0;
      if (this->GetInput()->GetSpacing()[dim] > maxSpacing)
        maxSpacing = this->GetInput()->GetSpacing()[dim];
      }
    }

  // Beta must be smaller than 1 / (2 ^ NumberOfDimensionsProcessed * max(spacing)) for the algorithm to converge
  m_Beta = 1/(pow(2,numberOfDimensionsProcessed) * maxSpacing)* 0.9 ;

  m_MultiplyFilter->SetConstant2(m_Beta);
  m_MagnitudeThresholdFilter->SetThreshold(m_Gamma);
  m_GradientFilter->SetDimensionsProcessed(m_DimensionsProcessed);
  m_DivergenceFilter->SetDimensionsProcessed(m_DimensionsProcessed);

  // Have the last filter calculate its output information,
  // which should update that of the whole pipeline
  m_MagnitudeThresholdFilter->UpdateOutputInformation();
  this->GetOutput()->CopyInformation( m_MultiplyFilter->GetOutput() );
}

template< typename TOutputImage, typename TGradientImage>
void
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::SetPipelineForFirstIteration()
{
  m_MultiplyFilter->SetInput1(this->GetInput());
  m_GradientFilter->SetInput(m_MultiplyFilter->GetOutput());
  m_MagnitudeThresholdFilter->SetInput(m_GradientFilter->GetOutput());
}

template< typename TOutputImage, typename TGradientImage>
void
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::SetPipelineAfterFirstIteration()
{
  m_SubtractFilter->SetInput1(this->GetInput());
  m_SubtractFilter->SetInput2(m_DivergenceFilter->GetOutput());

  m_MultiplyFilter->SetInput1(m_SubtractFilter->GetOutput());

  m_GradientFilter->SetInput(m_MultiplyFilter->GetOutput());

  m_SubtractGradientFilter->SetInput2(m_GradientFilter->GetOutput());

  m_MagnitudeThresholdFilter->SetInput(m_SubtractGradientFilter->GetOutput());
}

template< typename TOutputImage, typename TGradientImage>
void
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::GenerateData()
{
  typename TGradientImage::Pointer pimg;

  // The first iteration only updates intermediate variables, not the output
  // The output is updated m_NumberOfIterations-1 times, therefore an additional
  // iteration must be performed so that even m_NumberOfIterations=1 has an effect
  for (int iter=0; iter<m_NumberOfIterations; iter++)
    {
    if(iter==1) SetPipelineAfterFirstIteration();

    m_MagnitudeThresholdFilter->Update();

    pimg = m_MagnitudeThresholdFilter->GetOutput();
    pimg->DisconnectPipeline();
    m_DivergenceFilter->SetInput( pimg );
    m_SubtractGradientFilter->SetInput1( pimg );
    }
  m_DivergenceFilter->Update();
  pimg->ReleaseData();

  m_SubtractFilter->Update();
  this->GraftOutput(m_SubtractFilter->GetOutput());
}

} // end namespace rtk

#endif
