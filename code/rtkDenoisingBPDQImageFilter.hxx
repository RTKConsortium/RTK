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

#ifndef rtkDenoisingBPDQImageFilter_hxx
#define rtkDenoisingBPDQImageFilter_hxx

#include "rtkDenoisingBPDQImageFilter.h"

namespace rtk
{

template< typename TOutputImage, typename TGradientImage>
DenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::DenoisingBPDQImageFilter()
{
  m_Gamma = 1.0;
  m_NumberOfIterations = 1;
  m_MinSpacing = 0;

  // This is an InPlace filter only for the subclasses to have the possibility to run in place
  this->SetInPlace(false);

  // Create the sub filters
  m_GradientFilter = GradientFilterType::New();
  m_MultiplyFilter = MultiplyFilterType::New();
  m_SubtractFilter = SubtractImageFilterType::New();
  m_SubtractGradientFilter = SubtractGradientFilterType::New();
  m_DivergenceFilter = DivergenceFilterType::New();

  // Set whether the sub filters should release their data during pipeline execution
  m_DivergenceFilter->ReleaseDataFlagOn();
  m_SubtractFilter->ReleaseDataFlagOn(); // It is the pipeline's output, but it is explicitely computed during the last iteration
  m_GradientFilter->ReleaseDataFlagOn();
  m_MultiplyFilter->ReleaseDataFlagOn();
  m_SubtractGradientFilter->ReleaseDataFlagOn();

  // Set some filters to be InPlace

  // DenoisingBPDQ reaches its memory consumption peak
  // when m_SubtractGradientFilter allocates its output (a covariant vector image)
  // and uses its two inputs (two covariant vector images)
  // Setting it in place reduces the memory requirement from 3 covariant vector images to 2
  m_SubtractGradientFilter->SetInPlace(true);
}

template< typename TOutputImage, typename TGradientImage>
void
DenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::GenerateOutputInformation()
{
  // Set the pipeline for the first iteration
  SetPipelineForFirstIteration();

  // Compute the parameters used in Basis Pursuit Dequantization
  // and set the filters to use them
  double numberOfDimensionsProcessed = 0;
  m_MinSpacing = this->GetInput()->GetSpacing()[0];
  for (unsigned int dim=0; dim<TOutputImage::ImageDimension; dim++)
    {
    if (m_DimensionsProcessed[dim])
      {
      numberOfDimensionsProcessed += 1.0;
      if (this->GetInput()->GetSpacing()[dim] < m_MinSpacing)
        m_MinSpacing = this->GetInput()->GetSpacing()[dim];
      }
    }

  // Set the gradient and divergence filter to take spacing into account
  m_GradientFilter->SetUseImageSpacingOn();
  m_DivergenceFilter->SetUseImageSpacingOn();

  // Beta must be smaller than 1 / (2 ^ NumberOfDimensionsProcessed) for the algorithm to converge
  m_Beta = 1/pow(2,numberOfDimensionsProcessed) * 0.9 * m_MinSpacing;

  m_MultiplyFilter->SetConstant2(m_Beta);
  m_GradientFilter->SetDimensionsProcessed(m_DimensionsProcessed);
  m_DivergenceFilter->SetDimensionsProcessed(m_DimensionsProcessed);

  // Have the last filter calculate its output information,
  // which should update that of the whole pipeline
  this->GetThresholdFilter()->UpdateOutputInformation();
  this->GetOutput()->CopyInformation( m_MultiplyFilter->GetOutput() );
}

template< typename TOutputImage, typename TGradientImage>
void
DenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::SetPipelineForFirstIteration()
{
  m_MultiplyFilter->SetInput1(this->GetInput());
  m_GradientFilter->SetInput(m_MultiplyFilter->GetOutput());
  this->GetThresholdFilter()->SetInput(m_GradientFilter->GetOutput());
}

template< typename TOutputImage, typename TGradientImage>
void
DenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::SetPipelineAfterFirstIteration()
{
  m_SubtractFilter->SetInput1(this->GetInput());
  m_SubtractFilter->SetInput2(m_DivergenceFilter->GetOutput());

  m_MultiplyFilter->SetInput1(m_SubtractFilter->GetOutput());

  m_GradientFilter->SetInput(m_MultiplyFilter->GetOutput());

  m_SubtractGradientFilter->SetInput2(m_GradientFilter->GetOutput());

  this->GetThresholdFilter()->SetInput(m_SubtractGradientFilter->GetOutput());

  m_MultiplyFilter->SetConstant2(m_Beta * m_MinSpacing);
}

template< typename TOutputImage, typename TGradientImage>
void
DenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::GenerateData()
{
  typename TGradientImage::Pointer pimg;

  // The first iteration only updates intermediate variables, not the output
  // The output is updated m_NumberOfIterations-1 times, therefore an additional
  // iteration must be performed so that even m_NumberOfIterations=1 has an effect
  for (int iter=0; iter<m_NumberOfIterations; iter++)
    {
    if(iter==1) SetPipelineAfterFirstIteration();

    this->GetThresholdFilter()->Update();

    pimg = this->GetThresholdFilter()->GetOutput();
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
