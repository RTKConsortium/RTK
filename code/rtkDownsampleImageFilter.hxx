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

#ifndef rtkDownsampleImageFilter_hxx
#define rtkDownsampleImageFilter_hxx

#include "rtkDownsampleImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkObjectFactory.h"
#include "itkProgressReporter.h"

namespace rtk
{

/**
 *   Constructor
 */
template <class TInputImage, class TOutputImage>
DownsampleImageFilter<TInputImage,TOutputImage>
::DownsampleImageFilter()
{
  this->SetNumberOfRequiredInputs(1);
}

/**
 *
 */
template <class TInputImage, class TOutputImage>
void
DownsampleImageFilter<TInputImage,TOutputImage>
::SetFactors(unsigned int factors[])
{
  unsigned int j;

  this->Modified();
  for (j = 0; j < ImageDimension; j++)
    {
    m_Factors[j] = factors[j];
    if (m_Factors[j] < 1)
      {
      m_Factors[j] = 1;
      }
    }
}

/**
 *
 */
template <class TInputImage, class TOutputImage>
void
DownsampleImageFilter<TInputImage,TOutputImage>
::SetFactor(unsigned int dimension, unsigned int factor)
{
  unsigned int j;

  this->Modified();
  for (j = 0; j < ImageDimension; j++)
    {
    if (j == dimension)
      {
      m_Factors[j] = factor;
      }
    else
      {
      m_Factors[j] = 1;
      }
    }
}


//template <class TInputImage, class TOutputImage>
//void
//DownsampleImageFilter<TInputImage,TOutputImage>
//::BeforeThreadedGenerateData()
//{
//  std::cout << "In DownsampleImageFilter : BeforeThreadedGenerateData, input size = " << this->GetInput()->GetLargestPossibleRegion().GetSize() << std::endl;
//}

/**
 *
 */
template <class TInputImage, class TOutputImage>
void 
DownsampleImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType threadId)
{
  itkDebugMacro(<<"Actually executing");

  //Get the input and output pointers
  InputImageConstPointer  inputPtr    = this->GetInput();
  OutputImagePointer      outputPtr   = this->GetOutput();

  //Define/declare an iterator that will walk the output region for this
  //thread.
  typedef itk::ImageRegionIterator<TOutputImage>     OutputIterator;
  typedef itk::ImageRegionConstIterator<TInputImage> InputIterator;

  //Define a few indices that will be used to translate from an input pixel
  //to an output pixel
  typename TInputImage::IndexType inputStartIndex = inputPtr->GetLargestPossibleRegion().GetIndex();
  typename TOutputImage::IndexType outputStartIndex = outputPtr->GetLargestPossibleRegion().GetIndex();

  typename TOutputImage::OffsetType firstPixelOfOutputRegionForThreadOffset;
  typename TInputImage::OffsetType firstPixelOfInputRegionForThreadOffset;
  typename TOutputImage::OffsetType firstPixelOfOutputLineOffset;
  typename TInputImage::OffsetType firstPixelOfInputLineOffset;
  typename TOutputImage::OffsetType offset;

  //Support progress methods/callbacks
  itk::ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());

  //Unless the downsampling factor is 1, we always skip the first pixel
  //Create an offset array to enforce this behavior
  for (unsigned int i=0; i < TInputImage::ImageDimension; i++)
    {
    if (m_Factors[i] == 1)
      {
      offset[i] = 0;
      }
    else
      {
      offset[i] = 1;
      }
    }

  //Find the first input pixel that is copied to the output (the one with lowest indices
  //in all dimensions)
  firstPixelOfOutputRegionForThreadOffset =
      outputRegionForThread.GetIndex() - outputStartIndex;
  for (unsigned int dim=0; dim < TInputImage::ImageDimension; dim++)
    {
    firstPixelOfInputRegionForThreadOffset[dim] =
        firstPixelOfOutputRegionForThreadOffset[dim] * m_Factors[dim]
        + offset[dim];
    }

  // Walk the slice obtained by setting the first coordinate to zero.
  // Each pixel is the beginning of a line (a 1D vector traversing
  // the output region along the first dimension). For each pixel,
  // create an iterator and perform the copy from the input
  OutputImageRegionType slice = outputRegionForThread;
  slice.SetSize(0, 1);
  OutputIterator sliceIt(outputPtr, slice);

  while (!sliceIt.IsAtEnd())
    {
    //Determine the offset of the current pixel in the slice
    firstPixelOfOutputLineOffset = sliceIt.GetIndex() - outputStartIndex;

    //Calculate the offset of the corresponding input pixel
    for (unsigned int dim=0; dim < TInputImage::ImageDimension; dim++)
      {
      firstPixelOfInputLineOffset[dim] =
          firstPixelOfOutputLineOffset[dim] * m_Factors[dim]
          + offset[dim];
      }

    // Create the iterators
    typename TOutputImage::RegionType outputLine = outputRegionForThread;
    typename TOutputImage::SizeType outputLineSize;
    outputLineSize.Fill(1);
    outputLineSize[0] = outputRegionForThread.GetSize(0);
    outputLine.SetSize(outputLineSize);
    outputLine.SetIndex(sliceIt.GetIndex());

    typename TInputImage::RegionType inputLine = inputPtr->GetLargestPossibleRegion();
    typename TInputImage::SizeType inputLineSize;
    inputLineSize.Fill(1);

    // Short example of how to calculate the inputLineSize :
    // If we downsample by a factor 3 the vector [x a x x b x x],
    // (starting in "a" because we have already taken into account the
    // offset, using firstPixelOfInputLineOffset) we obtain [a b],
    // so all we need is a vector of length 4 = (2 - 1) * 3 + 1
    inputLineSize[0] = (outputLineSize[0] - 1) * m_Factors[0] + 1;
    inputLine.SetSize(inputLineSize);
    inputLine.SetIndex(inputStartIndex + firstPixelOfInputLineOffset);

    OutputIterator outIt(outputPtr, outputLine);
    InputIterator inIt(inputPtr, inputLine);

    // Walk the line and copy the pixels
    while(!outIt.IsAtEnd())
      {
      outIt.Set(inIt.Get());
      for (unsigned int i=0; i<m_Factors[0]; i++) ++inIt;
      ++outIt;

      progress.CompletedPixel();
      }

    // Move to next pixel in the slice
    ++sliceIt;
    }
}


//template <class TInputImage, class TOutputImage>
//void
//DownsampleImageFilter<TInputImage,TOutputImage>
//::AfterThreadedGenerateData()
//{
//  std::cout << "In DownsampleImageFilter : AfterThreadedGenerateData" << std::endl;
//}

/** 
 *
 */
template <class TInputImage, class TOutputImage>
void 
DownsampleImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion()
{
  //Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  //Get pointers to the input and output
  InputImagePointer  inputPtr  = const_cast<TInputImage *>(this->GetInput());
  OutputImagePointer outputPtr = this->GetOutput();

  if (!inputPtr || !outputPtr)
    {
    return;
    }
  inputPtr->SetRequestedRegionToLargestPossibleRegion();
}

/** 
 *
 */
template <class TInputImage, class TOutputImage>
void 
DownsampleImageFilter<TInputImage,TOutputImage>
::GenerateOutputInformation()
{
  //Call the superclass' implementation of this method
  Superclass::GenerateOutputInformation();

  //Get pointers to the input and output
  InputImageConstPointer  inputPtr  = this->GetInput();
  OutputImagePointer      outputPtr = this->GetOutput();

  if (!inputPtr || !outputPtr)
    {
    return;
    }

  //We need to compute the output spacing, the output image size, and the
  //output image start index
  unsigned int i;
  const typename TInputImage::SpacingType& inputSpacing = inputPtr->GetSpacing();
  const typename TInputImage::SizeType& inputSize = inputPtr->GetLargestPossibleRegion().GetSize();
  const typename TInputImage::IndexType& inputStartIndex = inputPtr->GetLargestPossibleRegion().GetIndex();

  typename TOutputImage::SpacingType  outputSpacing;
  typename TOutputImage::SizeType     outputSize;
  typename TOutputImage::IndexType    outputStartIndex;

  for (i = 0; i < TOutputImage::ImageDimension; i++)
    {
    outputSpacing[i] = inputSpacing[i] * (float)m_Factors[i];
    outputSize[i] = (unsigned long)floor((float) inputSize[i] / (float)m_Factors[i]);
    if (outputSize[i] < 1)
      {
      outputSize[i] = 1;
      }

    outputStartIndex[i] = (long)floor((float) inputStartIndex[i] / (float)m_Factors[i]);
    }

  outputPtr->SetSpacing(outputSpacing);

  typename TOutputImage::RegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion.SetSize(outputSize);
  outputLargestPossibleRegion.SetIndex(outputStartIndex);

  outputPtr->SetRegions(outputLargestPossibleRegion);
}

} // end namespace rtk

#endif
