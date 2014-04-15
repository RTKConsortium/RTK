/*=========================================================================

  Program:  GIFT Sample Activity Feature Generator
  Module:   giftSampleActivity.txx
  Language: C++
  Date:     2005/11/22
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftSampleActivity_TXX
#define __giftSampleActivity_TXX

//GIFT Includes
#include "giftSampleActivity.h"

//ITK includes
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"


namespace gift
{

/////////////////////////////////////////////////////////
//Constructor()
template <class TImage>
SampleActivity<TImage>
::SampleActivity()
{
    
}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void
SampleActivity<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}

                  
/////////////////////////////////////////////////////////
//GenerateData()
template <class TImage>
void
SampleActivity<TImage>
::GenerateData()
{
    //Get number of input and output images
//    unsigned int numberOfInputs = this->GetNumberOfRequiredInputs();
    unsigned int numberOfOutputs = this->GetNumberOfRequiredOutputs();

    //Allocate memory for outputs
    for (unsigned int idx = 0; idx < numberOfOutputs; idx++)
    {
        OutputImagePointer outputPtr = this->GetOutput(idx);
        outputPtr->SetBufferedRegion(outputPtr->GetRequestedRegion());
        outputPtr->Allocate();
    }

    //Graft outputs
    for (unsigned int index=0; index<numberOfOutputs; index++)
    {
        itk::DataObject* ptrOutputToGraft = this->itk::ProcessObject::GetInput(index);
        typename itk::ImageToImageFilter<TImage, TImage>::OutputImageType* outputToGraft = dynamic_cast<typename itk::ImageToImageFilter<TImage, TImage>::OutputImageType*>(ptrOutputToGraft);
        this->GraftNthOutput(index, outputToGraft); 
    }
}


}// end namespace gift

#endif
