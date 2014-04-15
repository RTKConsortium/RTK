/*=========================================================================

  Program:  GIFT Absolute Sample Activity Feature Generator
  Module:   giftAbsSampleActivitye.txx
  Language: C++
  Date:     2005/11/26
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftAbsSampleActivity_TXX
#define __giftAbsSampleActivity_TXX

//GIFT Includes
#include "giftAbsSampleActivity.h"

//ITK includes
#include "itkImage.h"
#include "itkAbsImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"


namespace gift
{

/////////////////////////////////////////////////////////
//Constructor()
template <class TImage>
AbsSampleActivity<TImage>
::AbsSampleActivity()
{
    
}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void
AbsSampleActivity<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}

                  
/////////////////////////////////////////////////////////
//GenerateData()
template <class TImage>
void
AbsSampleActivity<TImage>
::GenerateData()
{
    //Get number of input and output images
//    unsigned int numberOfInputs = this->GetNumberOfRequiredInputs();
    unsigned int numberOfOutputs = this->GetNumberOfRequiredOutputs();

    //Foreach output
    for (unsigned int index = 0; index < numberOfOutputs; index++)
    {
        //Allocate pointer
        OutputImagePointer outputPtr = this->GetOutput(index);
        outputPtr->SetBufferedRegion(outputPtr->GetRequestedRegion());
        outputPtr->Allocate();

        //Attach input to AbsImageFilter
        typedef itk::AbsImageFilter<TImage, TImage> AbsFilterType;
        typename AbsFilterType::Pointer filterAbs = AbsFilterType::New();
        filterAbs->SetInput(this->GetInput(index));
        filterAbs->Update();

        //Graft to output
        this->GraftNthOutput(index, filterAbs->GetOutput());    
    }
}


}// end namespace gift

#endif
