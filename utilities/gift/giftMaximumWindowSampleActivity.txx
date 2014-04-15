/*=========================================================================

  Program:  GIFT Maximum Window Sample Feature Generator
  Module:   giftMaximumWindowSampleActivity.h
  Language: C++
  Date:     2006/06/03
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2006 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftMaximumWindowSampleActivity_TXX
#define __giftMaximumWindowSampleActivity_TXX

//GIFT Includes
#include "giftMaximumWindowSampleActivity.h"

//ITK includes
#include "itkImage.h"
#include "itkAbsImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkMaximumWindowImageFilter.h"

namespace gift
{

/////////////////////////////////////////////////////////
//Constructor()
template <class TImage>
MaximumWindowSampleActivity<TImage>
::MaximumWindowSampleActivity()
{

}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void
MaximumWindowSampleActivity<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}

                  
/////////////////////////////////////////////////////////
//GenerateData()
template <class TImage>
void
MaximumWindowSampleActivity<TImage>
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

        //Compute maximum of window
        typedef itk::MaximumWindowImageFilter<TImage, TImage> MaxWindowFilterType;
        typename MaxWindowFilterType::Pointer filterMaxWindow = MaxWindowFilterType::New();
        filterMaxWindow->SetInput(filterAbs->GetOutput());
        filterMaxWindow->SetRadius(m_WindowSize);
        filterMaxWindow->Update();

        //Graft to output
        this->GraftNthOutput(index, filterMaxWindow->GetOutput());
    }
}


}// end namespace gift

#endif
