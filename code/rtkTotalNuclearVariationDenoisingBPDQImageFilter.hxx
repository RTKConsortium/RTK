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

#ifndef rtkTotalNuclearVariationDenoisingBPDQImageFilter_hxx
#define rtkTotalNuclearVariationDenoisingBPDQImageFilter_hxx

#include "rtkTotalNuclearVariationDenoisingBPDQImageFilter.h"

namespace rtk
{

template< typename TOutputImage, typename TGradientImage>
TotalNuclearVariationDenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::TotalNuclearVariationDenoisingBPDQImageFilter()
{
  // Default behaviour is to process all spatial dimensions, i.e. all but the last one, which contains channels
  for (unsigned int dim = 0; dim < TOutputImage::ImageDimension - 1; dim++)
    {
    this->m_DimensionsProcessed[dim] = true;
    }
  this->m_DimensionsProcessed[TOutputImage::ImageDimension - 1] = false;

  // Create the SingularValueThresholdFilter to replace the generic filter
  // used in the base class
  this->m_ThresholdFilter = SingularValueThresholdFilterType::New();
}

template< typename TOutputImage, typename TGradientImage>
void
TotalNuclearVariationDenoisingBPDQImageFilter<TOutputImage, TGradientImage>
::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();

  this->m_ThresholdFilter->SetThreshold(this->m_Gamma);
}

} // end namespace rtk

#endif
