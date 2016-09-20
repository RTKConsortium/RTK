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
#ifndef rtkAverageOutOfROIImageFilter_hxx
#define rtkAverageOutOfROIImageFilter_hxx

#include "rtkAverageOutOfROIImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
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

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  // Set the direction along which the output requested region should NOT be split
  m_Splitter = itk::ImageRegionSplitterDirection::New();
  m_Splitter->SetDirection(TInputImage::ImageDimension - 1);
#else
  // Old versions of ITK (before 4.4) do not have the ImageRegionSplitterDirection
  // and should run this filter with only one thread
  this->SetNumberOfThreads(1);
#endif
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

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  template< class TInputImage, class TROI >
  const itk::ImageRegionSplitterBase*
  AverageOutOfROIImageFilter< TInputImage, TROI >
  ::GetImageRegionSplitter(void) const
  {
    return m_Splitter;
  }
#endif

template< class TInputImage, class TROI >
void
AverageOutOfROIImageFilter< TInputImage, TROI >
::BeforeThreadedGenerateData()
{
#if !(ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4))
  if (this->GetNumberOfThreads() > 1)
    {
    itkWarningMacro(<< "AverageOutOfROIImageFilter cannot use multiple threads with ITK versions older than v4.4. Reverting to single thread behavior");
    this->SetNumberOfThreads(1);
    }
#endif
}

template< class TInputImage, class TROI >
void
AverageOutOfROIImageFilter< TInputImage, TROI >
::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();

  // Check whether the ROI has the same information as the input image
  typename TROI::SizeType ROISize = this->GetROI()->GetLargestPossibleRegion().GetSize();
  typename TROI::SpacingType ROISpacing = this->GetROI()->GetSpacing();
  typename TROI::PointType ROIOrigin = this->GetROI()->GetOrigin();
  typename TROI::DirectionType ROIDirection = this->GetROI()->GetDirection();

  bool isInformationInconsistent;
  isInformationInconsistent = false;

  for (unsigned int dim=0; dim<TROI::ImageDimension; dim++)
    {
    if (ROISize[dim] != this->GetInput(0)->GetLargestPossibleRegion().GetSize()[dim])
      isInformationInconsistent = true;
    if (ROISpacing[dim] != this->GetInput(0)->GetSpacing()[dim])
      isInformationInconsistent = true;
    if (ROIOrigin[dim] != this->GetInput(0)->GetOrigin()[dim])
      isInformationInconsistent = true;
    for (unsigned int i=0; i<TROI::ImageDimension; i++)
      {
      if (ROIDirection(dim, i) != this->GetInput(0)->GetDirection()(dim, i))
        isInformationInconsistent = true;
      }
    }

  if(isInformationInconsistent)
    itkGenericExceptionMacro(<< "In AverageOutOfROIImageFilter: information of ROI image does not match input image");

  this->GetOutput()->SetLargestPossibleRegion(this->GetInput(0)->GetLargestPossibleRegion());
}

template< class TInputImage, class TROI >
void
AverageOutOfROIImageFilter< TInputImage, TROI >
::GenerateInputRequestedRegion()
{
  //Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // Compute the requested regions on input and ROI from the output's requested region
  typename TInputImage::RegionType outputRequested = this->GetOutput()->GetRequestedRegion();

  typename TInputImage::RegionType inputRequested = outputRequested;
  inputRequested.SetSize(TInputImage::ImageDimension - 1, this->GetInput(0)->GetLargestPossibleRegion().GetSize(TInputImage::ImageDimension - 1));
  inputRequested.SetIndex(TInputImage::ImageDimension - 1, this->GetInput(0)->GetLargestPossibleRegion().GetIndex(TInputImage::ImageDimension - 1));

  typename TROI::RegionType ROIRequested;
//  for (unsigned int dim = 0; dim<TROI::ImageDimension; dim++)
//    {
//    ROIRequested.SetSize(dim, outputRequested.GetSize(dim));
//    ROIRequested.SetIndex(dim, outputRequested.GetSize(dim));
//    }
  ROIRequested = outputRequested.Slice(TInputImage::ImageDimension - 1);

  //Get pointers to the input and ROI
  typename TInputImage::Pointer  inputPtr  = const_cast<TInputImage *>(this->GetInput(0));
  inputPtr->SetRequestedRegion(inputRequested);

  typename TROI::Pointer  ROIPtr  = this->GetROI();
  ROIPtr->SetRequestedRegion(ROIRequested);
}

template< class TInputImage, class TROI >
void
AverageOutOfROIImageFilter< TInputImage, TROI >
::ThreadedGenerateData(const typename TInputImage::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  // Walks the first frame of the outputRegionForThread
  // For each voxel, creates an input iterator that walks
  // the last dimension and computes the average, and
  // a similar output iterator that replaces the output
  // by their average along last dimension if ROI=1

  // Create a region containing only the first frame of outputRegionForThread
  typename TInputImage::RegionType FirstFrameRegion = outputRegionForThread;
  FirstFrameRegion.SetSize(TInputImage::ImageDimension - 1, 1);

  // Create a similar region with TROI image type (last dimension removed)
  typename TROI::RegionType SingleFrameRegion = this->GetROI()->GetLargestPossibleRegion();
  for (unsigned int dim = 0; dim< TInputImage::ImageDimension - 1; dim++)
    {
    SingleFrameRegion.SetSize(dim, FirstFrameRegion.GetSize(dim));
    SingleFrameRegion.SetIndex(dim, FirstFrameRegion.GetIndex(dim));
    }

  // Iterate on these regions
  itk::ImageRegionIteratorWithIndex<TInputImage> FakeIterator(this->GetOutput(), FirstFrameRegion);
  itk::ImageRegionIterator<TROI> ROIIterator(this->GetROI(), SingleFrameRegion);

  // Create a single-voxel region traversing last dimension
  typename TInputImage::RegionType SingleVoxelRegion = outputRegionForThread;
  for (unsigned int dim = 0; dim< TInputImage::ImageDimension - 1; dim++)
      SingleVoxelRegion.SetSize(dim, 1);

  // Create a variable to store the average value
  typename TInputImage::PixelType avg;

  while(!ROIIterator.IsAtEnd())
    {
    // Configure the SingleVoxelRegion correctly to follow the FakeIterator
    // It is the only purpose of this FakeIterator
    SingleVoxelRegion.SetIndex(FakeIterator.GetIndex());

    // Walk the input along last dimension for this voxel, averaging along the way
    itk::ImageRegionConstIterator<TInputImage> inputIterator(this->GetInput(), SingleVoxelRegion);
    avg = 0;
    while (!inputIterator.IsAtEnd())
      {
      avg += inputIterator.Get();
      ++inputIterator;
      }
    avg /= SingleVoxelRegion.GetSize(TInputImage::ImageDimension - 1);

    // Walk the output along last dimension for this voxel,
    // replacing voxel values with their average, if indicated by the value in ROI
    itk::ImageRegionIterator<TInputImage> outputIterator(this->GetOutput(), SingleVoxelRegion);
    while (!outputIterator.IsAtEnd())
      {
      outputIterator.Set(ROIIterator.Get() * outputIterator.Get() + avg * (1 - ROIIterator.Get()));
      ++outputIterator;
      }

    ++FakeIterator;
    ++ROIIterator;
    }
}

} // end namespace itk

#endif
