/*=========================================================================

  Program:  GIFT Average Feature Weight Generator
  Module:   giftAverageFeature.txx
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
#ifndef __giftAverageFeature_TXX
#define __giftAverageFeature_TXX

//GIFT Includes
#include "giftAverageFeature.h"

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
AverageFeature<TImage>
::AverageFeature()
{

}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void
AverageFeature<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}

                  
/////////////////////////////////////////////////////////
//GenerateData()
template <class TImage>
void
AverageFeature<TImage>
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

        //// Clear the content of the output
        //currentOutputIterator.GoToBegin();
        //while (!currentOutputIterator.IsAtEnd())
        //{
        //  currentOutputIterator.Set(itk::NumericTraits<OutputPixelType>::Zero);
        //  ++currentOutputIterator;
        //}
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
        ////Sum all feature values
        //itk::NumericTraits<InputPixelType>::AccumulateType sum = itk::NumericTraits<InputPixelType>::Zero;
        //for (unsigned int indexFeature=0; indexFeature<this->GetNumberOfFeatureMaps(); indexFeature++)
        //{
        //  sum += featureIterators[indexFeature].Get();
        //}
        
        //Set weight map
        for (unsigned int index=0; index<this->GetNumberOfWeightMaps(); index++)
        {
            //Set the weight to be an average of the number of images
            //Eg. 1/2 = 0.5, 1/3 = 0.333, 1/4 = 0.25, etc...
            outputIterators[index].Set(1.0/this->GetNumberOfFeatureMaps());
        }
        
        //Increment output iterators
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
