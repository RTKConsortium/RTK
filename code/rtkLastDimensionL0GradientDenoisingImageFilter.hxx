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
#ifndef rtkLastDimensionL0GradientDenoisingImageFilter_hxx
#define rtkLastDimensionL0GradientDenoisingImageFilter_hxx

#include "rtkLastDimensionL0GradientDenoisingImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"

namespace rtk
{
//
// Constructor
//
template< class TInputImage >
LastDimensionL0GradientDenoisingImageFilter< TInputImage >
::LastDimensionL0GradientDenoisingImageFilter()
{
#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  // Set the direction along which the output requested region should NOT be split
  m_Splitter = itk::ImageRegionSplitterDirection::New();
  m_Splitter->SetDirection(TInputImage::ImageDimension - 1);
#else
  // Old versions of ITK (before 4.4) do not have the ImageRegionSplitterDirection
  // and should run this filter with only one thread
  this->SetNumberOfThreads(1);
#endif

  // Default parameters
  m_NumberOfIterations = 5;
  m_Lambda = 0.;
}

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
template< class TInputImage >
const itk::ImageRegionSplitterBase*
LastDimensionL0GradientDenoisingImageFilter< TInputImage >
::GetImageRegionSplitter(void) const
{
  return m_Splitter;
}
#endif

template< class TInputImage >
void
LastDimensionL0GradientDenoisingImageFilter< TInputImage >
::BeforeThreadedGenerateData()
{
#if !(ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4))
  if (this->GetNumberOfThreads() > 1)
    {
    itkWarningMacro(<< "LastDimensionL0GradientDenoisingImageFilter cannot use multiple threads with ITK versions older than v4.4. Reverting to single thread behavior");
    this->SetNumberOfThreads(1);
    }
#endif
}

template< class TInputImage >
void
LastDimensionL0GradientDenoisingImageFilter< TInputImage >
::GenerateInputRequestedRegion()
{
  //Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // Initialize the input requested region to the output requested region, then extend it along the last dimension
  typename TInputImage::RegionType inputRequested = this->GetOutput()->GetRequestedRegion();
  inputRequested.SetSize(TInputImage::ImageDimension - 1, this->GetInput(0)->GetLargestPossibleRegion().GetSize(TInputImage::ImageDimension - 1));
  inputRequested.SetIndex(TInputImage::ImageDimension - 1, this->GetInput(0)->GetLargestPossibleRegion().GetIndex(TInputImage::ImageDimension - 1));

  //Get pointers to the input and ROI
  typename TInputImage::Pointer  inputPtr  = const_cast<TInputImage *>(this->GetInput());
  inputPtr->SetRequestedRegion(inputRequested);
}

template< class TInputImage >
void
LastDimensionL0GradientDenoisingImageFilter< TInputImage >
::OneDimensionMinimizeL0NormOfGradient(InputPixelType* input, unsigned int length, double lambda, unsigned int nbIters)
{
  // Initialize
  float beta = 0;
  std::vector<unsigned int> firsts(length);
  std::vector<float> weights(length);
  std::vector<InputPixelType> values(length);
  unsigned int nbGroups = length;

  // Fill in the arrays
  for (unsigned int i=0; i < length; i++)
    {
    firsts[i] = i;
    weights[i] = 1.0;
    values[i] = input[i];
    }
  
  // Main loop
  for (unsigned int iter=0; iter<nbIters; iter++)
    {
    // Set the threshold for the current iteration (beta increases over time towards lambda)
    beta = (float) iter / (float) nbIters * lambda;  
      
    // Run through all groups  
    unsigned int i=0;
    while (i < nbGroups - 1)
      {
      // Check the right neighbour of the current group
      unsigned int j = (i + 1) % nbGroups;
    
      // Decide whether or not to merge the current group with its right neighbour
      if ( weights[i] * weights[j] * (values[i] - values[j]) * (values[i] - values[j]) <= beta * (weights[i] + weights[j]) )
        {
        // Perform the fusion
          
        // The merged group's value is the weighted mean between the values of both groups
        values[i] = ( weights[i] * values[i] + weights[j] * values[j] ) / (weights[i] + weights[j]);
          
        // The merged group's weight is the sum of the weights of both groups
        weights[i] = weights[i] + weights[j];
      
        // Remove j-th element by shifting everything after it
        for (unsigned int k=j+1; k<length; k++)
          {
          firsts[k-1] = firsts[k];
          values[k-1] = values[k];
          weights[k-1] = weights[k];
          }
        firsts[length-1] = 0;
        values[length-1] = 0;
        weights[length-1] = 0;
        
        // Decrement the total number of groups
        nbGroups--;
        }
        
      // Move to next group
      i++;
      }
    }
    
  // Assemble the pieces to create the denoised output (overwriting the input)
  for (unsigned int i=0; i<nbGroups; i++)
    {
    for (unsigned int j=firsts[i]; j< firsts[i] + weights[i]; j++)
      {
      input[ j % length ] = values[i];
      }
    }
}

template< class TInputImage >
void
LastDimensionL0GradientDenoisingImageFilter< TInputImage >
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

  // Iterate on this region
  itk::ImageRegionIteratorWithIndex<TInputImage> FakeIterator(this->GetOutput(), FirstFrameRegion);

  // Create a single-voxel region traversing last dimension
  typename TInputImage::RegionType SingleVoxelRegion = outputRegionForThread;
  for (unsigned int dim = 0; dim< TInputImage::ImageDimension - 1; dim++)
    {
    SingleVoxelRegion.SetSize(dim, 1);
    SingleVoxelRegion.SetIndex(dim, 0);
    }

  while(!FakeIterator.IsAtEnd())
    {
    // Configure the SingleVoxelRegion correctly to follow the FakeIterator
    // It is the only purpose of this FakeIterator
    SingleVoxelRegion.SetIndex(FakeIterator.GetIndex());

    // Walk the input along last dimension for this voxel, filling an array with the values read
    itk::ImageRegionConstIterator<TInputImage> inputIterator(this->GetInput(), SingleVoxelRegion);
    InputPixelType* toBeRegularized = new InputPixelType[outputRegionForThread.GetSize(TInputImage::ImageDimension - 1)];
  
    unsigned int i=0;
    while (!inputIterator.IsAtEnd())
      {
      toBeRegularized[i] = inputIterator.Get();
      i++;
      ++inputIterator;
      }
  
    // Perform regularization (in place) on this array
    OneDimensionMinimizeL0NormOfGradient(toBeRegularized, outputRegionForThread.GetSize(TInputImage::ImageDimension - 1), this->GetLambda(), this->GetNumberOfIterations());
  
    // Walk the output along last dimension for this voxel,
    // replacing voxel with their regularized value
    itk::ImageRegionIterator<TInputImage> outputIterator(this->GetOutput(), SingleVoxelRegion);
    i=0;
    while (!outputIterator.IsAtEnd())
      {
      outputIterator.Set(toBeRegularized[i]);
      i++;
      ++outputIterator;
      }

    delete [] toBeRegularized;
      
    ++FakeIterator;
    }
}

} // end namespace itk

#endif
