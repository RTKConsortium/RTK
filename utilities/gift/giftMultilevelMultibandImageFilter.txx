/*=========================================================================

  Program:  GIFT Multi-level Multi-band Image Filter
  Module:   giftMultilevelMultibandImageFilter.txx
  Language: C++
  Date:     2005/11/22
  Version:  0.2
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftMultilevelMultibandImageFilter_TXX
#define __giftMultilevelMultibandImageFilter_TXX

//Includes
#include "giftMultilevelMultibandImageFilter.h"

namespace gift
{

/////////////////////////////////////////////////////////
//Default Constructor
template <class TImage>
MultilevelMultibandImageFilter<TImage>::MultilevelMultibandImageFilter()
{
    //Initialise private variables
    this->m_NumberOfInputImages     = 0;
    this->m_NumberOfOutputImages    = 0;
    this->m_NumberOfInputLevels     = 0;
    this->m_NumberOfOutputLevels    = 0;
    this->m_NumberOfInputBands      = 0;
    this->m_NumberOfOutputBands     = 0;
    this->m_Type = Self::User;
}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void MultilevelMultibandImageFilter<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}


/////////////////////////////////////////////////////////
//ModifyInputOutputStorage()
template <class TImage>
void MultilevelMultibandImageFilter<TImage>
::ModifyInputOutputStorage()
{
    //Set as modified
    this->Modified();

    //Set required number of inputs/outputs
    unsigned int requiredNumberOfInputs = this->CalculateNumberOfInputs();
    this->SetNumberOfRequiredInputs(requiredNumberOfInputs);
    int requiredNumberOfOutputs = this->CalculateNumberOfOutputs();
    this->SetNumberOfRequiredOutputs(requiredNumberOfOutputs);

    //Make acutal outputs and required outputs match
//    unsigned int actualNumberOfOutputs = static_cast<unsigned int>(this->GetNumberOfOutputs());
    int actualNumberOfOutputs = this->GetNumberOfOutputs();
    int idx;

    if (actualNumberOfOutputs < requiredNumberOfOutputs)
    {
        //Add extra outputs
        for (idx = actualNumberOfOutputs; idx < requiredNumberOfOutputs; idx++)
        {
            typename itk::DataObject::Pointer output = this->MakeOutput(idx);
            this->SetNthOutput(idx, output.GetPointer());
        }
    }
    else if (actualNumberOfOutputs > requiredNumberOfOutputs)
    {
        //Remove extra outputs
        for (idx = (actualNumberOfOutputs-1); idx >= requiredNumberOfOutputs; idx--)
        {
            if (idx < 0){break;}

            typename itk::DataObject::Pointer output = this->GetOutputs()[idx];
//            this->RemoveOutput(output);
        }
    }
}


/////////////////////////////////////////////////////////
//GetOutputByImageLevelBand()
template <class TImage>
typename MultilevelMultibandImageFilter<TImage>::OutputImageType *
MultilevelMultibandImageFilter<TImage>
::GetOutputByImageLevelBand(unsigned int image, unsigned int level, unsigned int band)
{
    int index = this->ConvertOutputImageLevelBandToIndex(image, level, band);
    return this->GetOutput(index);
}

/////////////////////////////////////////////////////////
//GetDataObjectOutputByImageLevelBand()
template <class TImage>
itk::DataObject *
MultilevelMultibandImageFilter<TImage>
::GetDataObjectOutputByImageLevelBand(unsigned int image, unsigned int level, unsigned int band)
{
    int index = this->ConvertOutputImageLevelBandToIndex(image, level, band);
    return this->GetOutput(index);
}


/////////////////////////////////////////////////////////
//SetInputByImageLevelBand()
template <class TImage>
void
MultilevelMultibandImageFilter<TImage>
::SetInputByImageLevelBand(unsigned int image, unsigned int level, unsigned int band, const InputImageType * input)
{
    int index = this->ConvertInputImageLevelBandToIndex(image, level, band);
    this->SetInput(index, input);
}


/////////////////////////////////////////////////////////
//GetInputByImageLevelBand()
template <class TImage>
const typename MultilevelMultibandImageFilter<TImage>::InputImageType *
MultilevelMultibandImageFilter<TImage>
::GetInputByImageLevelBand(unsigned int image, unsigned int level, unsigned int band)
{
    int index = this->ConvertInputImageLevelBandToIndex(image, level, band);
    return this->GetInput(index);
}


/////////////////////////////////////////////////////////
//CalculateNumberOfInputs()
template <class TImage>
unsigned int MultilevelMultibandImageFilter<TImage>::CalculateNumberOfInputs()
{
    switch(this->m_Type)
    {
    case Self::Deconstruction:
        return this->GetNumberOfInputImages();
    case Self::Reconstruction:
        //Fall through
    case Self::Merge:
        //Fall through
    case Self::User:
        return (this->GetNumberOfInputImages()*
                this->GetNumberOfInputLevels()*
                this->GetNumberOfInputBands());
    default:
        return 0;
    }//end switch(Type)
};


/////////////////////////////////////////////////////////
//CalculateNumberOfOutputs()
template <class TImage>
unsigned int MultilevelMultibandImageFilter<TImage>::CalculateNumberOfOutputs()
{
    switch(this->m_Type)
    {
    case Self::Reconstruction:
        return this->GetNumberOfOutputImages();
    case Self::Deconstruction:
        //Fall through
    case Self::Merge:
        //Fall through
    case Self::User:
        return (this->GetNumberOfOutputImages()*
                this->GetNumberOfOutputLevels()*
                this->GetNumberOfOutputBands());
    default:
        return 0;
    }//end switch(Type)
};


/////////////////////////////////////////////////////////
//ConvertInputIndexToImageLevelBand()
template <class TImage>
void MultilevelMultibandImageFilter<TImage>
::ConvertInputIndexToImageLevelBand(unsigned int index, 
                                    unsigned int &image,
                                    unsigned int &level, 
                                    unsigned int &band)
{
    image = index/(this->GetNumberOfInputLevels()*this->GetNumberOfInputBands());
    unsigned int indexForImage = index - (image*this->GetNumberOfInputLevels()*this->GetNumberOfInputBands());
    level = indexForImage/this->GetNumberOfInputBands();
    unsigned int indexForLevel = indexForImage - (level*this->GetNumberOfInputBands());
    band = indexForLevel;
};


/////////////////////////////////////////////////////////
//ConvertOutputIndexToImageLevelBand()
template <class TImage>
void MultilevelMultibandImageFilter<TImage>
::ConvertOutputIndexToImageLevelBand(unsigned int index, 
                                     unsigned int &image,
                                     unsigned int &level, 
                                     unsigned int &band)
{
    image = index/(this->GetNumberOfOutputLevels()*this->GetNumberOfOutputBands());
    unsigned int indexForImage = index - (image*this->GetNumberOfOutputLevels()*this->GetNumberOfOutputBands());
    level = indexForImage/this->GetNumberOfOutputBands();
    unsigned int indexForLevel = indexForImage - (level*this->GetNumberOfOutputBands());
    band = indexForLevel;
};


/////////////////////////////////////////////////////////
//ConvertInputImageLevelBandToIndex()
template <class TImage>
unsigned int MultilevelMultibandImageFilter<TImage>
::ConvertInputImageLevelBandToIndex(unsigned int image,
                                    unsigned int level, 
                                    unsigned int band)
{
    //NOTE: Level 0 is the first level...
    unsigned int index = image*this->GetNumberOfInputLevels()*this->GetNumberOfInputBands() + 
                         level*this->GetNumberOfInputBands() +
                         band;
    return index;
};


/////////////////////////////////////////////////////////
//ConvertOutputImageLevelBandToIndex()
template <class TImage>
unsigned int MultilevelMultibandImageFilter<TImage>
::ConvertOutputImageLevelBandToIndex(unsigned int image,
                                     unsigned int level, 
                                     unsigned int band)
{
    //NOTE: Level 0 is the first level...
    unsigned int index = image*this->GetNumberOfOutputLevels()*this->GetNumberOfOutputBands() + 
                         level*this->GetNumberOfOutputBands() +
                         band;
    return index;
};


/////////////////////////////////////////////////////////
//GenerateInputRequestedRegion()
template <class TImage>
void MultilevelMultibandImageFilter<TImage>
::GenerateInputRequestedRegion()
{
    //Get first input
    itk::ImageBase<ImageDimension> *firstInput;  
    firstInput = const_cast<InputImageType*>(static_cast<const InputImageType*>(this->GetInput(0)));
    
    //Set each requested region to be the largest possible region
    for (unsigned int index=0; index < this->GetNumberOfInputs(); ++index)
    {
        //Get current input
        itk::ImageBase<ImageDimension> *currentInput;  
        currentInput = const_cast<InputImageType*>(static_cast<const InputImageType*>(this->GetInput(index)));

        typename itk::ImageBase<ImageDimension>::RegionType currentRegion = currentInput->GetRequestedRegion();
        currentRegion.Crop(currentInput->GetLargestPossibleRegion());
        currentInput->SetRequestedRegion(currentRegion);

    }//end for
}


/////////////////////////////////////////////////////////
//GenerateOutputInformation()
template <class TImage>
void MultilevelMultibandImageFilter<TImage>
::GenerateOutputInformation()
{
    //NOTE: We assume that NumberOfInputImages = NumberOfOutputImages.
    //      Special subclasses of MultilevelMultibandImageFilter must override
    //      this method to change this default behaviour...

    itk::DataObject::Pointer input;
    itk::DataObject::Pointer output;

    //Copy the information to the output from each corresponding input
//    for (unsigned int index=0; index<this->GetNumberOfRequiredInputs(); index++)
    for (unsigned int index=0; index<this->GetNumberOfInputImages(); index++)
    {
        //Get the input and output at this index
        input = const_cast<InputImageType*>(static_cast<const InputImageType*>(this->GetInput(index)));
        output = this->GetOutput(index);

        if (input && output)
        {
            //Copy the input information to the output
            output->CopyInformation(input);

            //Also copy the requested region
            output->SetRequestedRegion(input);
        }//end if
    }//end for
}


/////////////////////////////////////////////////////////
//GenerateOutputRequestedRegion()
template <class TImage>
void MultilevelMultibandImageFilter<TImage>
::GenerateOutputRequestedRegion(itk::DataObject *output)
{
    //Set each requested region to be the largest possible region
    for (unsigned int index=0; index < this->GetNumberOfOutputs(); ++index)
    {
        itk::ImageBase<ImageDimension> *givenData;  
        itk::ImageBase<ImageDimension> *currentData;  
        givenData = dynamic_cast<itk::ImageBase<ImageDimension>*>(output);
        currentData = dynamic_cast<itk::ImageBase<ImageDimension>*>(this->GetOutput(index));

        typename itk::ImageBase<ImageDimension>::RegionType currentRegion = currentData->GetRequestedRegion();
        currentRegion.Crop(currentData->GetLargestPossibleRegion());
        currentData->SetRequestedRegion(currentRegion);
    }//end for      
}


}// end namespace gift

#endif
