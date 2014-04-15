/*=========================================================================

  Program:  GIFT Median Window Sample Feature Generator
  Module:   giftMedianWindowSampleActivity.h
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
#ifndef __giftMedianWindowSampleActivity_TXX
#define __giftMedianWindowSampleActivity_TXX

//GIFT Includes
#include "giftMedianWindowSampleActivity.h"

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
MedianWindowSampleActivity<TImage>
::MedianWindowSampleActivity()
{

}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void
MedianWindowSampleActivity<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}

                  
/////////////////////////////////////////////////////////
//GenerateData()
template <class TImage>
void
MedianWindowSampleActivity<TImage>
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

        //Attach AbsImageFilter to MedianFilter
        typename InternalMedianFilterType::Pointer filterMedian = InternalMedianFilterType::New();
        filterMedian->SetInput(filterAbs->GetOutput());
        filterMedian->Update();

        //Graft to output
        this->GraftNthOutput(index, filterMedian->GetOutput());
    }
}


}// end namespace gift

#endif
