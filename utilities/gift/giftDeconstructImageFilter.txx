/*=========================================================================

  Program:  GIFT Multi-level Multi-band Image Filter
  Module:   giftDeconstructImageFilter.txx
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
#ifndef __giftDeconstructImageFilter_TXX
#define __giftDeconstructImageFilter_TXX

//Includes
#include "giftDeconstructImageFilter.h"

namespace gift
{

/////////////////////////////////////////////////////////
//Default Constructor
template <class TImage>
DeconstructImageFilter<TImage>::DeconstructImageFilter()
{
  //Initialise private variables
  this->m_NumberOfLevels     = 0;

  // Create single filter
  m_PadFilter = PadFilterType::New();
}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void DeconstructImageFilter<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}


/////////////////////////////////////////////////////////
//ModifyInputOutputStorage()
template <class TImage>
void DeconstructImageFilter<TImage>
::ModifyInputOutputStorage()
{
    //Set as modified
    this->Modified();

    //Set required number of outputs
    unsigned int requiredNumberOfOutputs = this->CalculateNumberOfOutputs();
    this->SetNumberOfRequiredOutputs(requiredNumberOfOutputs);

    //Make actual outputs and required outputs match
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
            this->RemoveOutput(output);
        }
    }
}


template <class TImage>
typename DeconstructImageFilter<TImage>::OutputImageType *
DeconstructImageFilter<TImage>
::GetOutputByLevelBand(unsigned int level, unsigned int band)
{
    int index = this->ConvertOutputLevelBandToIndex(level, band);
    return this->GetOutput(index);
}

template <class TImage>
itk::DataObject *
DeconstructImageFilter<TImage>
::GetDataObjectOutputByLevelBand(unsigned int level, unsigned int band)
{
    int index = this->ConvertOutputLevelBandToIndex(level, band);
    return this->GetOutput(index);
}

template <class TImage>
unsigned int
DeconstructImageFilter<TImage>
::GetNumberOfOutputBands(unsigned int level){
  if(level==0) return floor(pow(2, TImage::Dimension) + 0.5);
  else return floor(pow(2, TImage::Dimension) - 0.5);
}

template <class TImage>
unsigned int DeconstructImageFilter<TImage>::CalculateNumberOfOutputs()
{
  unsigned int numberOfOutputs = 0;
  for (unsigned int level=0; level<m_NumberOfLevels; level++){
      numberOfOutputs += this->GetNumberOfOutputBands(level);
    }
  return numberOfOutputs;
}

template <class TImage>
void DeconstructImageFilter<TImage>
::ConvertOutputIndexToLevelBand(unsigned int index,
                                     unsigned int &level, 
                                     unsigned int &band)
{
  unsigned int currentLevel = 0;
  unsigned int currentIndex = index;
  while (index > this->GetNumberOfOutputBands(currentLevel))
    {
    currentIndex -= this->GetNumberOfOutputBands(currentLevel);
    currentLevel++;
    }
  level = currentLevel;
  band = currentIndex;
}

template <class TImage>
unsigned int DeconstructImageFilter<TImage>
::ConvertOutputImageLevelBandToIndex(unsigned int level,
                                     unsigned int band)
{
  unsigned int index = band;
  for (unsigned int l=0; l<level; l++){
      index += this->GetNumberOfOutputBands(l);
    }
}

template <class TImage>
typename InputImageType::Pointer
unsigned int DeconstructImageFilter<TImage>
::ComputeNdKernel(unsigned int index)
{

}

template <class TImage>
void DeconstructImageFilter<TImage>
::GenerateOutputInformation()
{
  //Create the pipeline
  for (unsigned int level=m_NumberOfLevels; level>=0; level++)
    {
    for (unsigned int band=0; band<this->GetNumberOfOutputBands(level); band++)
      {

      }
    }

}

template <class TImage>
void DeconstructImageFilter<TImage>
::GenerateData()
{
  //TODO : implement using the pipeline
}


}// end namespace gift

#endif
