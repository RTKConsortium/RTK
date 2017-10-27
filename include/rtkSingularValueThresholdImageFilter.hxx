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

#ifndef rtkSingularValueThresholdImageFilter_hxx
#define rtkSingularValueThresholdImageFilter_hxx

#include "rtkSingularValueThresholdImageFilter.h"

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include <itkProgressReporter.h>

namespace rtk
{

template< typename TInputImage, typename TRealType, typename TOutputImage >
SingularValueThresholdImageFilter< TInputImage, TRealType, TOutputImage >
::SingularValueThresholdImageFilter()
{
  m_Threshold = 0;

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

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  template< typename TInputImage, typename TRealType, typename TOutputImage >
  const itk::ImageRegionSplitterBase*
  SingularValueThresholdImageFilter< TInputImage, TRealType, TOutputImage >
  ::GetImageRegionSplitter(void) const
  {
    return m_Splitter;
  }
#endif

  template< typename TInputImage, typename TRealType, typename TOutputImage >
  void
  SingularValueThresholdImageFilter< TInputImage, TRealType, TOutputImage >
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

template< typename TInputImage, typename TRealType, typename TOutputImage >
void
SingularValueThresholdImageFilter< TInputImage, TRealType, TOutputImage >
::ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread,
                       itk::ThreadIdType threadId)
{
  // Walks the first frame of the outputRegionForThread
  // For each voxel, creates an input iterator that walks
  // the last dimension and computes the average, and
  // a similar output iterator that replaces the output
  // by their average along last dimension if ROI=1

  // Support progress methods/callbacks
  itk::ProgressReporter progress( this, threadId, outputRegionForThread.GetNumberOfPixels() );

  // Create a region containing only the first frame of outputRegionForThread
  typename TInputImage::RegionType FirstFrameRegion = outputRegionForThread;
  FirstFrameRegion.SetSize(TInputImage::ImageDimension - 1, 1);

  // Iterate on these regions
  itk::ImageRegionIteratorWithIndex<TInputImage> FakeIterator(this->GetOutput(), FirstFrameRegion);

  // Create a single-voxel region traversing last dimension
  typename TInputImage::RegionType SingleVoxelRegion = outputRegionForThread;
  for (unsigned int dim = 0; dim< TInputImage::ImageDimension - 1; dim++)
      SingleVoxelRegion.SetSize(dim, 1);

  // Create a variable to store the jacobian matrix
  vnl_matrix<double> jacobian (outputRegionForThread.GetSize()[TInputImage::ImageDimension - 1], TInputImage::ImageDimension - 1);

  while(!FakeIterator.IsAtEnd())
    {
    // Configure the SingleVoxelRegion correctly to follow the FakeIterator
    // It is the only purpose of this FakeIterator
    SingleVoxelRegion.SetIndex(FakeIterator.GetIndex());

    // Walk the input along last dimension for this voxel, filling in the jacobian along the way
    itk::ImageRegionConstIterator<TInputImage> inputIterator(this->GetInput(), SingleVoxelRegion);
    unsigned int row=0;
    for (inputIterator.GoToBegin(); !inputIterator.IsAtEnd(); ++inputIterator, row++)
      {
      typename TInputImage::PixelType gradient = inputIterator.Get();
      for (unsigned int column=0; column<TInputImage::ImageDimension - 1; column++)
        {
        jacobian[row][column] = gradient[column];
        }
      }

    // Perform the singular value decomposition of the jacobian matrix
    vnl_svd<double> svd(jacobian);

    // Threshold the singular values. Since they are sorted in descending order, we
    // can stop as soon as we reach one that is below threshold
    for (unsigned int i=0; (i<TInputImage::ImageDimension && svd.W(i) > m_Threshold); i++)
      {
      svd.W(i) = m_Threshold;
      }

    // Reconstruct the jacobian
    jacobian = svd.recompose();

    // Walk the output along last dimension for this voxel,
    // replacing gradient vectors with their newly computed value
    itk::ImageRegionIterator<TInputImage> outputIterator(this->GetOutput(), SingleVoxelRegion);
    row = 0;
    for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator, row++)
      {
      typename TInputImage::PixelType vector;
      for (unsigned int column=0; column<jacobian.cols(); column++)
        vector[column] = jacobian[row][column];

      outputIterator.Set(vector);
      }

    ++FakeIterator;
    progress.CompletedPixel();
    }
}

} // end namespace itk

#endif
