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

#ifndef rtkUpsampleImageFilter_hxx
#define rtkUpsampleImageFilter_hxx

#include "rtkUpsampleImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkObjectFactory.h"
#include "itkProgressReporter.h"
#include "itkNumericTraits.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
UpsampleImageFilter<TInputImage,TOutputImage>
::UpsampleImageFilter()
{
  this->SetNumberOfRequiredInputs(1);
  this->m_Order = 0;
  this->m_OutputSize.Fill(0);
  this->m_OutputIndex.Fill(0);
}

template <class TInputImage, class TOutputImage>
void
UpsampleImageFilter<TInputImage,TOutputImage>
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

template <class TInputImage, class TOutputImage>
void
UpsampleImageFilter<TInputImage,TOutputImage>
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

template <class TInputImage, class TOutputImage>
void 
UpsampleImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType threadId)
{
  //Get the input and output pointers
  InputImageConstPointer  inputPtr    = this->GetInput();
  OutputImagePointer      outputPtr   = this->GetOutput();

  //Define/declare an iterator that will walk the output region for this
  //thread.
  typedef itk::ImageRegionIterator<TOutputImage>      OutputIterator;
  typedef itk::ImageRegionConstIterator<TInputImage>  InputIterator;
  OutputIterator outIt(outputPtr, outputRegionForThread);

  //Fill the output region with zeros
  while (!outIt.IsAtEnd())
    {
    outIt.Set(itk::NumericTraits<typename TOutputImage::PixelType>::Zero);
    ++outIt;
    }

  //Define a few indices that will be used to translate from an input pixel
  //to an output pixel
  typename TOutputImage::IndexType  outputStartIndex;
  typename TInputImage::IndexType   inputStartIndex;

  typename TInputImage::OffsetType  inputOffset;
  typename TOutputImage::OffsetType firstValidPixelOffset;
  typename TOutputImage::OffsetType firstPixelOfLineOffset;

  outputStartIndex = outputPtr->GetLargestPossibleRegion().GetIndex();
  inputStartIndex = inputPtr->GetLargestPossibleRegion().GetIndex();

  //Support progress methods/callbacks
  itk::ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());

  //Find the first output pixel that is copied from the input (the one with lowest indices
  //in all dimensions)
  firstValidPixelOffset = outputRegionForThread.GetIndex() - outputStartIndex;
  for (unsigned int i=0; i<TOutputImage::ImageDimension; i++)
    {
    while((firstValidPixelOffset[i]-1) % m_Factors[i])
      {
      firstValidPixelOffset[i] = firstValidPixelOffset[i]+1;
      }
    }

  // Walk the slice obtained by setting the first coordinate to zero. If the
  // line (1D vector traversing the output region along the first dimension)
  // contains pixels that should be copied from the input,
  // create an iterator and perform the copies
  OutputImageRegionType slice = outputRegionForThread;
  slice.SetSize(0, 1);
  slice.SetIndex(0, outputRegionForThread.GetIndex(0) + firstValidPixelOffset[0]);

  OutputIterator sliceIt(outputPtr, slice);
  while (!sliceIt.IsAtEnd())
    {
    //Determine the offset of the current pixel in the slice
    firstPixelOfLineOffset = sliceIt.GetIndex() - outputStartIndex;

    //Check whether the line contains pixels that should be copied from the input
    bool copyFromInput = true;
    for (unsigned int dim=0; dim < TInputImage::ImageDimension; dim++)
      {
      if ((firstPixelOfLineOffset[dim]-1) % m_Factors[dim])
        copyFromInput = false;
      }

    // If it does, create an iterator along the line and copy the pixels
    if (copyFromInput)
      {
      //Calculate the corresponding input index
      for (unsigned int i=0; i<TOutputImage::ImageDimension; i++)
        {
        inputOffset[i] = (firstPixelOfLineOffset[i]-1) / m_Factors[i];
        }

      // Create the iterators
      typename TOutputImage::RegionType outputLine = slice;
      typename TOutputImage::SizeType outputLineSize;
      outputLineSize.Fill(1);
      outputLineSize[0] = outputRegionForThread.GetSize(0) - firstPixelOfLineOffset[0];
      outputLine.SetSize(outputLineSize);
      outputLine.SetIndex(sliceIt.GetIndex());

      typename TInputImage::RegionType inputLine = inputPtr->GetLargestPossibleRegion();
      typename TInputImage::SizeType inputLineSize;
      inputLineSize.Fill(1);
      inputLineSize[0] = (outputLineSize[0]+1) / m_Factors[0];
      inputLine.SetSize(inputLineSize);
      inputLine.SetIndex(inputStartIndex + inputOffset);

      OutputIterator outIt(outputPtr, outputLine);
      InputIterator inIt(inputPtr, inputLine);

      // Walk the line and copy the pixels
      while(!inIt.IsAtEnd())
        {
        outIt.Set(inIt.Get());
        for (unsigned int i=0; i<m_Factors[0]; i++) ++outIt;
        ++inIt;

        progress.CompletedPixel();
        }

      }
    // Move to next pixel in the slice
    ++sliceIt;
    }
}

template <class TInputImage, class TOutputImage>
void 
UpsampleImageFilter<TInputImage,TOutputImage>
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

template <class TInputImage, class TOutputImage>
void 
UpsampleImageFilter<TInputImage,TOutputImage>
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

  typename TOutputImage::SpacingType  outputSpacing;
  typename TOutputImage::SizeType     outputSize;
  typename TOutputImage::IndexType    outputStartIndex;

  for (i = 0; i < TOutputImage::ImageDimension; i++)
    {
    outputSpacing[i] = inputSpacing[i] / (double)m_Factors[i];
    outputSize[i] = m_OutputSize[i];
    outputStartIndex[i] = m_OutputIndex[i]+1;
    }

  outputPtr->SetSpacing(outputSpacing);

  typename TOutputImage::RegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion.SetSize(outputSize);
  outputLargestPossibleRegion.SetIndex(outputStartIndex);

  outputPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);
  outputPtr->SetOrigin(inputPtr->GetOrigin());
}

} // end namespace rtk

#endif
