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

#ifndef rtkTotalVariationDenoisingBPDQImageFilter_hxx
#define rtkTotalVariationDenoisingBPDQImageFilter_hxx

#include "rtkTotalVariationDenoisingBPDQImageFilter.h"

namespace rtk
{

template< typename TOutputImage, typename TGradientImage>
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::TotalVariationDenoisingBPDQImageFilter()
{
  // Default behaviour is to process all dimensions
  for (unsigned int dim = 0; dim < TOutputImage::ImageDimension; dim++)
    {
    this->m_DimensionsProcessed[dim] = true;
    }

  // Create the sub filters
  m_ThresholdFilter = MagnitudeThresholdFilterType::New();
}

template< typename TOutputImage, typename TGradientImage>
void
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::SetDimensionsProcessed(bool* arg)
{
  bool Modified=false;
  for (unsigned int dim=0; dim<TOutputImage::ImageDimension; dim++)
    {
    if (this->m_DimensionsProcessed[dim] != arg[dim])
      {
      this->m_DimensionsProcessed[dim] = arg[dim];
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
  this->m_GradientFilter->OverrideBoundaryCondition(new itk::PeriodicBoundaryCondition<TOutputImage>());
  this->m_DivergenceFilter->OverrideBoundaryCondition(new itk::PeriodicBoundaryCondition<TGradientImage>());
}

template< typename TOutputImage, typename TGradientImage>
void
TotalVariationDenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();

  this->m_ThresholdFilter->SetThreshold(this->m_Gamma);
}

} // end namespace rtk

#endif
