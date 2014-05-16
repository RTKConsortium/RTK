/*=========================================================================
 *
 *  Copyright Insight Software Consortium
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
#ifndef __itkAverageOutOfROIImageFilter_txx
#define __itkAverageOutOfROIImageFilter_txx

#include "rtkAverageOutOfROIImageFilter.h"
#include "itkImageRegionIterator.h"


namespace rtk
{
//
// Constructor
//
template< class TInputImage, class TROI >
AverageOutOfROIImageFilter< TInputImage, TROI >
::AverageOutOfROIImageFilter()
{
  this->SetNumberOfRequiredInputs(2);
}

template< class TInputImage, class TROI >
void
AverageOutOfROIImageFilter< TInputImage, TROI >
::SetROI(const TROI* Map)
{
  this->SetNthInput(1, const_cast<TROI*>(Map));
}

template< class TInputImage, class TROI >
typename TROI::Pointer
AverageOutOfROIImageFilter< TInputImage, TROI >
::GetROI()
{
  return static_cast< TROI* >
          ( this->itk::ProcessObject::GetInput(1) );
}

template< class TInputImage, class TROI >
void
AverageOutOfROIImageFilter< TInputImage, TROI >
::GenerateData()
{
  int Dimension = this->GetInput()->GetImageDimension();
  int NumberPhases = this->GetInput()->GetLargestPossibleRegion().GetSize()[Dimension-1];

  this->GetOutput()->SetRegions(this->GetInput()->GetLargestPossibleRegion());
  this->GetOutput()->Allocate();

  typename AccumulateFilterType::Pointer accumulateFilter = AccumulateFilterType::New();
  accumulateFilter->SetInput(this->GetInput());
  accumulateFilter->AverageOn();
  accumulateFilter->SetAccumulateDimension(Dimension-1);
  accumulateFilter->Update();

  typename TInputImage::RegionType SinglePhaseRegion = accumulateFilter->GetOutput()->GetLargestPossibleRegion();
  typename TInputImage::RegionType AccumulateRegion = accumulateFilter->GetOutput()->GetLargestPossibleRegion();
  typename TROI::RegionType ROILargest = this->GetROI()->GetLargestPossibleRegion();

  for (int PhaseNumber=0; PhaseNumber<NumberPhases; PhaseNumber++)
    {
    itk::ImageRegionIterator<TInputImage> accumulateIterator(accumulateFilter->GetOutput(), AccumulateRegion);

    typename TInputImage::IndexType SinglePhaseRegionIndex = SinglePhaseRegion.GetIndex();
    SinglePhaseRegionIndex[Dimension-1] = PhaseNumber;
    SinglePhaseRegion.SetIndex(SinglePhaseRegionIndex);

    itk::ImageRegionIterator<TInputImage> outputIterator(this->GetOutput(), SinglePhaseRegion);
    itk::ImageRegionConstIterator<TInputImage> inputIterator(this->GetInput(), SinglePhaseRegion);
    itk::ImageRegionIterator<TROI> ROIIterator(this->GetROI(), ROILargest);

    while(!ROIIterator.IsAtEnd())
      {
        if (ROIIterator.Get()) outputIterator.Set( accumulateIterator.Get() );
        else outputIterator.Set(0);

      ++outputIterator;
      ++accumulateIterator;
      ++inputIterator;
      ++ROIIterator;
      }
    }
}

} // end namespace itk

#endif
