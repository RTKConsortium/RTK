/*=========================================================================

  Program:  GIFT Linear Weight Combiner
  Module:   giftLinearWeightCombiner.txx
  Language: C++
  Date:     2005/11/23
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftLinearWeightCombiner_TXX
#define __giftLinearWeightCombiner_TXX

//GIFT Includes
#include "giftLinearWeightCombiner.h"

//ITK includes
#include "itkImageRegionIterator.h"
#include "itkNumericTraits.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkShiftScaleImageFilter.h"

namespace gift
{

/////////////////////////////////////////////////////////
//Constructor()
template <class TImage>
LinearWeightCombiner<TImage>
::LinearWeightCombiner()
{

}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void
LinearWeightCombiner<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}

                  
/////////////////////////////////////////////////////////
//GenerateData()
template <class TImage>
void
LinearWeightCombiner<TImage>
::GenerateData()
{
    //NOTE: We assume the inputs to this filter as as follows 
    //      (where input[x] is a multi-level/multi-band image...)
    //input[0] = image0
    //input[1] = image1
    //input[n] = imageN
    //input[n+1] = weightImage0
    //input[n+2] = weightImage1
    //input[n+n] = weightImageN

    //Loop through each level and band
    for (unsigned int level=0; level<this->GetNumberOfInputLevels(); level++)
    {
        for (unsigned int band=0; band<this->GetNumberOfInputBands(); band++)
        {
            //Firstly allocate each output buffer
            unsigned int indexOutput = this->ConvertOutputImageLevelBandToIndex(0, level, band);
            OutputImagePointer outputPtr = this->GetOutput(indexOutput);
            outputPtr->SetBufferedRegion(outputPtr->GetRequestedRegion());
            outputPtr->Allocate();

            //Get output iterator
            itk::ImageRegionIterator<TImage> outputIt(outputPtr, outputPtr->GetLargestPossibleRegion());

            // Clear the content of the output
            outputIt.GoToBegin();
            while (!outputIt.IsAtEnd())
            {
                outputIt.Set(itk::NumericTraits<PixelType>::Zero);
                ++outputIt;
            }

            //Foreach input and weightInput
            unsigned int numberOfImageToFuse = this->GetNumberOfInputImages()/2;
            for (unsigned int image=0; image<(numberOfImageToFuse); image++)
            {
                //Get actual input pointers
                unsigned int inputActualIndex = this->ConvertInputImageLevelBandToIndex(image, level, band);
                InputImagePointer ptrInputActual = 
                  const_cast<TImage*>(static_cast<const InputImageType*>(this->GetInput(inputActualIndex)));

                //Get input weight iterator
                unsigned int inputWeightIndex = this->ConvertInputImageLevelBandToIndex(image+numberOfImageToFuse, level, band);
                InputImagePointer ptrInputWeight = 
                    const_cast<TImage*>(static_cast<const InputImageType*>(this->GetInput(inputWeightIndex)));

                //If both pointers are valid...
                if (ptrInputActual && ptrInputWeight)
                {
                    //Get iterators
                    itk::ImageRegionIterator<TImage> inputActualIt(ptrInputActual, ptrInputActual->GetLargestPossibleRegion());
                    itk::ImageRegionIterator<TImage> inputWeightIt(ptrInputWeight, ptrInputWeight->GetLargestPossibleRegion());

                    //Do weight for this pixel
                    inputActualIt.GoToBegin();
                    inputWeightIt.GoToBegin();
                    outputIt.GoToBegin();
                    while (!outputIt.IsAtEnd()) 
                    {
                        //outputIt.Set( m_Functor( outputIt.Get(), inputIt.Get() ) );
                        outputIt.Set(outputIt.Get() + (inputActualIt.Get()*inputWeightIt.Get()));
                        ++inputActualIt;
                        ++inputWeightIt;
                        ++outputIt;

                    }//end while (output iterator is not at end)
                }//end if (pointers valid)

            }//end foreach image
        }//end foreach band
    }//end foreach level
}

}// end namespace gift

#endif
