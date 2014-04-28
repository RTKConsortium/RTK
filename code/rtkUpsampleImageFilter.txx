/*=========================================================================

  Program:  rtk (Generalised Image Fusion Toolkit)
  Module:   rtkUpsampleImageFilter.txx
  Language: C++
  Date:     2005/11/16
  Version:  1.0
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See rtkCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
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


/**
 *
 */
template <class TInputImage, class TOutputImage>
void 
UpsampleImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
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

//    //Support progress methods/callbacks
//    itk::ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());
    
    typename TOutputImage::OffsetType outputStartIndex;
    for (unsigned int i=0; i<TOutputImage::ImageDimension; i++)
      {
      outputStartIndex[i] =   outputPtr->GetLargestPossibleRegion().GetIndex(i);
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
            inputIndex[dim] = floor( (double)(outputIndex[dim]-1) / (double)m_Factors[dim] );

            //Check within bounds
            if (inputIndex[dim] > ((int)inputSize[dim] - 1))
            {
                copyFromInput &= false;
            }
        }

        if (copyFromInput)
        {
            //Copy the output value from the input
            outIt.Set(inputPtr->GetPixel(inputIndex));
        }
        else
        {
            //Copy zero to the output value
            outIt.Set(itk::NumericTraits<typename TOutputImage::PixelType>::Zero);
        }

        //Increment iterator and progress reporter
        ++outIt;
//        progress.CompletedPixel();
    }
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
//        inputRequestedRegionSize[i]
//        = (long) ceil((double)outputRequestedRegionSize[i] /
//                      (double) m_Factors[i]);

//        inputRequestedRegionStartIndex[i]
//        = (long) floor((double)outputRequestedRegionStartIndex[i] /
//                       (double)m_Factors[i]);
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
    const typename TInputImage::SizeType& inputSize = inputPtr->GetLargestPossibleRegion().GetSize();
    const typename TInputImage::IndexType&    inputStartIndex = inputPtr->GetLargestPossibleRegion().GetIndex();

    typename TOutputImage::SpacingType  outputSpacing;
    typename TOutputImage::SizeType     outputSize;
    typename TOutputImage::IndexType    outputStartIndex;

    typename TInputImage::OffsetType offset;
  
    for (i = 0; i < TOutputImage::ImageDimension; i++)
    {
        outputSpacing[i] = inputSpacing[i] / (double)m_Factors[i];
        outputSize[i] = inputSize[i] * (unsigned long) m_Factors[i] + 1;
        offset[i] = -(int)m_Order +1;
    }
  
    outputPtr->SetSpacing(outputSpacing);
    outputStartIndex = inputStartIndex + offset;
  
    typename TOutputImage::RegionType outputLargestPossibleRegion;
    outputLargestPossibleRegion.SetSize(outputSize);
    outputLargestPossibleRegion.SetIndex(outputStartIndex);
  
    outputPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);
}

} // end namespace rtk

#endif
