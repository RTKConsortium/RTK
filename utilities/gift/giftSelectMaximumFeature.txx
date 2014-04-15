/*=========================================================================

  Program:  GIFT Select Maximum Feature Weight Generator
  Module:   giftSelectMaximumFeature.txx
  Language: C++
  Date:     2005/11/24
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftSelectMaximumFeature_TXX
#define __giftSelectMaximumFeature_TXX

//GIFT Includes
#include "giftSelectMaximumFeature.h"

//ITK includes
#include "itkImageRegionIterator.h"
#include "itkNumericTraits.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"


namespace gift
{

/////////////////////////////////////////////////////////
//Constructor()
template <class TImage>
SelectMaximumFeature<TImage>
::SelectMaximumFeature()
{

}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void
SelectMaximumFeature<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}

                  
/////////////////////////////////////////////////////////
//GenerateData()
template <class TImage>
void
SelectMaximumFeature<TImage>
::GenerateData()
{

    //NOTE: We have assumed that NumberOfFeatureMaps may NOT equal NumberOfWeightMaps 
    
    //Setup iterator containers
    typedef itk::ImageRegionIterator<TImage> IteratorType;
    std::vector<IteratorType> outputIterators;
    std::vector<IteratorType> featureIterators;
    outputIterators.resize(0);
    featureIterators.resize(0);

    //foreach output
    for (unsigned int indexOutputs=0; indexOutputs<this->GetNumberOfWeightMaps(); indexOutputs++)
    {
        //Allocate each output buffer
        OutputImagePointer outputPtr = this->GetOutput(indexOutputs);
        outputPtr->SetBufferedRegion(outputPtr->GetRequestedRegion());
        outputPtr->Allocate();

        //Get output iterator
        itk::ImageRegionIterator<TImage> currentOutputIterator(outputPtr, outputPtr->GetLargestPossibleRegion());
        currentOutputIterator.GoToBegin();
        outputIterators.push_back(currentOutputIterator);
    }

    //foreach feature
    for (unsigned int feature=0; feature<this->GetNumberOfFeatureMaps(); feature++)
    {
        //Get pointer
        InputImagePointer ptrFeatureMap = 
                  const_cast<TImage*>(static_cast<const InputImageType*>(this->GetInput(feature)));
        
        //Get feature iterator
        itk::ImageRegionIterator<TImage> currentFeatureIterator(ptrFeatureMap, ptrFeatureMap->GetLargestPossibleRegion());
        currentFeatureIterator.GoToBegin();
        featureIterators.push_back(currentFeatureIterator);
    }//end foreach feature

    //Cycle through each input (feature) and output (weight) map
    while (!outputIterators[0].IsAtEnd()) 
    {
        //Find maximum
        unsigned int indexFeatureMax = 0;
        PixelType maxValue = itk::NumericTraits<PixelType>::NonpositiveMin();
        
        for (unsigned featureIndex=0; featureIndex<this->GetNumberOfFeatureMaps(); featureIndex++)
        {
            PixelType currentValue = featureIterators[featureIndex].Get(); 
            if (currentValue > maxValue)
            {
                maxValue = currentValue;
                indexFeatureMax = featureIndex;
            }
        }

        //Set weight map
        for (unsigned int indexWeight=0; indexWeight<this->GetNumberOfWeightMaps(); indexWeight++)
        {
            if (indexWeight == indexFeatureMax)
            {
                //This is the maximum
                outputIterators[indexWeight].Set(1.0);
            }
            else
            {
                //This is NOT the maximum
                outputIterators[indexWeight].Set(0.0);
            }
        }
        
        //Increment input iterators
        for (unsigned int index=0; index<this->GetNumberOfWeightMaps(); index++)
        {
            ++outputIterators[index];       
        }

        //Increment input iterators
        for (unsigned int index=0; index<this->GetNumberOfFeatureMaps(); index++)
        {
            ++featureIterators[index];  
        }

    }//end while (output iterator is not at end)
}


}// end namespace gift

#endif
