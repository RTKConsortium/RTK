/*=========================================================================

  Program:  GIFT (Generalised Image Fusion Toolkit)
  Module:   giftDownsampleImageFilter.txx
  Language: C++
  Date:     2005/11/16
  Version:  1.0
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftDownsampleImageFilter_TXX
#define __giftDownsampleImageFilter_TXX

#include "giftDownsampleImageFilter.h"

#include "itkImageRegionIterator.h"
#include "itkObjectFactory.h"
#include "itkProgressReporter.h"

namespace gift
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
    typedef itk::ImageRegionIterator<TOutputImage> OutputIterator;
    OutputIterator outIt(outputPtr, outputRegionForThread);
  
    //Define a few indices that will be used to translate from an input pixel
    //to an output pixel
    typename TOutputImage::IndexType outputIndex;
    typename TInputImage::IndexType inputIndex;
    typename TOutputImage::SizeType factor;
    typename TOutputImage::OffsetType offset;

    // Get the input image start index (not necessarily zero because of padding)
    typename TInputImage::OffsetType inputStartIndex;
    typename TInputImage::OffsetType outputStartIndex;
    for (unsigned int i=0; i < TInputImage::ImageDimension; i++)
    {
        inputStartIndex[i] = inputPtr->GetRequestedRegion().GetIndex(i);
        outputStartIndex[i] = outputPtr->GetLargestPossibleRegion().GetIndex(i);
    }

    //Generate factor and offset array
    for (unsigned int i=0; i < TInputImage::ImageDimension; i++)
    {
        factor[i] = m_Factors[i];
        if (factor[i] == 1)
        {
            offset[i] = 0;
        }
        else
        {
            offset[i] = 1;
        }
    }
  
    //Support progress methods/callbacks
    itk::ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());
    
    //Walk the output region, and sample the input image
    while (!outIt.IsAtEnd())
    {
        //Determine the index of the output pixel
        outputIndex = outIt.GetIndex() - outputStartIndex;

        //Determine the input pixel location associated with this output pixel
        inputIndex = (outputIndex * factor) + inputStartIndex + offset;

        //Copy the input pixel to the output
        outIt.Set(inputPtr->GetPixel(inputIndex));
        ++outIt;

        progress.CompletedPixel();
    }
}


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
  
//    //We need to compute the input requested region (size and start index)
//    unsigned int i;
//    const typename TOutputImage::SizeType& outputRequestedRegionSize
//        = outputPtr->GetRequestedRegion().GetSize();
//    const typename TOutputImage::IndexType& outputRequestedRegionStartIndex
//        = outputPtr->GetRequestedRegion().GetIndex();
  
//    typename TInputImage::SizeType  inputRequestedRegionSize;
//    typename TInputImage::IndexType inputRequestedRegionStartIndex;
  
//    for (i = 0; i < TInputImage::ImageDimension; i++)
//    {
//        inputRequestedRegionSize[i] = outputRequestedRegionSize[i] * m_Factors[i];
//        inputRequestedRegionStartIndex[i] = outputRequestedRegionStartIndex[i] * (long)m_Factors[i];
//    }
  
//    typename TInputImage::RegionType inputRequestedRegion;
//    inputRequestedRegion.SetSize(inputRequestedRegionSize);
//    inputRequestedRegion.SetIndex(inputRequestedRegionStartIndex);

//    inputRequestedRegion.Crop(inputPtr->GetLargestPossibleRegion());
  
//    inputPtr->SetRequestedRegion(inputRequestedRegion);

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
    
//        outputStartIndex[i] = (long)ceil((float) inputStartIndex[i] / (float)m_Factors[i]);
        outputStartIndex[i] = 0;
    }
  
    outputPtr->SetSpacing(outputSpacing);
  
    typename TOutputImage::RegionType outputLargestPossibleRegion;
    outputLargestPossibleRegion.SetSize(outputSize);
    outputLargestPossibleRegion.SetIndex(outputStartIndex);

    outputPtr->SetRegions(outputLargestPossibleRegion);
}

} // end namespace gift

#endif
