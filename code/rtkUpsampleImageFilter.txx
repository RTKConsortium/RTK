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

#ifndef __rtkUpsampleImageFilter_TXX
#define __rtkUpsampleImageFilter_TXX

#include "rtkUpsampleImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkObjectFactory.h"
#include "itkProgressReporter.h"
#include "itkNumericTraits.h"

namespace rtk
{

/**
 *   Constructor
 */
template <class TInputImage, class TOutputImage>
UpsampleImageFilter<TInputImage,TOutputImage>
::UpsampleImageFilter()
{
  this->SetNumberOfRequiredInputs(1);
  this->m_Order = 0;
}

/**
 *
 */
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

/**
 *
 */
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
::BeforeThreadedGenerateData()
{
  std::cout << "In UpsampleImageFilter : BeforeThreadedGenerateData, input size = " << this->GetInput()->GetLargestPossibleRegion().GetSize() << std::endl;
}


/**
 *
 */
template <class TInputImage, class TOutputImage>
void 
UpsampleImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  double eps = 10e-6;

  //Get the input and output pointers
  InputImageConstPointer  inputPtr    = this->GetInput();
  OutputImagePointer      outputPtr   = this->GetOutput();

  //Get input size
  const typename TInputImage::SizeType& inputSize = inputPtr->GetLargestPossibleRegion().GetSize();

  //Define/declare an iterator that will walk the output region for this
  //thread.
  typedef itk::ImageRegionIterator<TOutputImage> OutputIterator;
  OutputIterator outIt(outputPtr, outputRegionForThread);

  //Define a few indices that will be used to translate from an input pixel
  //to an output pixel
  typename TOutputImage::IndexType outputIndex;
  typename TInputImage::IndexType inputIndex;

  typename TOutputImage::OffsetType outputStartIndex;
  typename TOutputImage::OffsetType inputStartIndex;
  for (unsigned int i=0; i<TOutputImage::ImageDimension; i++)
    {
    outputStartIndex[i] =   outputPtr->GetLargestPossibleRegion().GetIndex(i);
    inputStartIndex[i] =  inputPtr->GetLargestPossibleRegion().GetIndex(i);
    }

  //Walk the output region, and sample the input image
  while (!outIt.IsAtEnd())
    {
    //Determine the index of the output pixel
    outputIndex = outIt.GetIndex() - outputStartIndex;

    //Determine if the current output pixel is zero OR an input value
    bool copyFromInput = true;
    for (unsigned int dim=0; dim < TInputImage::ImageDimension; dim++)
      {
      if ((outputIndex[dim]-1) % m_Factors[dim])
        {
        copyFromInput = false;
        }

      //Caculate inputIndex (note: will not be used if copyFromInput=false...)
      inputIndex[dim] = round( (double)(outputIndex[dim]-1) / (double)m_Factors[dim] + eps);

      //Check within bounds
      if (inputIndex[dim] > ((int)inputSize[dim] - 1))
        {
        copyFromInput &= false;
        }
      }

    if (copyFromInput)
      {
      //Copy the output value from the input
      outIt.Set(inputPtr->GetPixel(inputIndex + inputStartIndex));
      }
    else
      {
      //Copy zero to the output value
      outIt.Set(itk::NumericTraits<typename TOutputImage::PixelType>::Zero);
      }

    //Increment iterator and progress reporter
    ++outIt;
    }
}

template <class TInputImage, class TOutputImage>
void
UpsampleImageFilter<TInputImage,TOutputImage>
::AfterThreadedGenerateData()
{
  std::cout << "In UpsampleImageFilter : AfterThreadedGenerateData" << std::endl;
}

/** 
 *
 */
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

/** 
 *
 */
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
