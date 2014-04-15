/*=========================================================================

  Program:  GIFT Weight Generator
  Module:   giftWeightGenerator.txx
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
#ifndef __giftWeightGenerator_TXX
#define __giftWeightGenerator_TXX

//GIFT Includes
#include "giftWeightGenerator.h"

//ITK includes

namespace gift
{

/////////////////////////////////////////////////////////
//Constructor()
template <class TImage>
WeightGenerator<TImage>
::WeightGenerator()
{   
    //Resize FeatureGeneratorContainer
    this->m_FeaturesToUse.resize(0);
    
    //Set the number of levels and bands
    this->SetNumberOfInputLevels(this->GetNumberOfInputLevels());
    this->SetNumberOfOutputLevels(this->GetNumberOfOutputLevels());
    this->SetNumberOfInputBands(this->GetNumberOfInputBands());
    this->SetNumberOfOutputBands(this->GetNumberOfOutputBands());
}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void
WeightGenerator<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}


/////////////////////////////////////////////////////////
//IsFeatureToBeUsed()
template <class TImage>
bool
WeightGenerator<TImage>
::IsFeatureToBeUsed(FeatureGeneratorPointer feature)
{
    typename FeatureGeneratorContainer::iterator itFeatureGenerator;

    //Iterate through features to be used list
    for (itFeatureGenerator = m_FeaturesToUse.begin(); 
        itFeatureGenerator != m_FeaturesToUse.end(); 
        itFeatureGenerator++)
    {
        if (*itFeatureGenerator == feature)
        {
            //Found the feature in our list
            return true;
        }
    }//end for

    //Did not feature - we don't need it
    return false;
}


}// end namespace gift

#endif
