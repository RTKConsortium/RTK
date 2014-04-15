/*=========================================================================

  Program:  GIFT Feature Generator
  Module:   giftFeatureGenerator.txx
  Language: C++
  Date:     2005/11/27
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftFeatureGenerator_TXX
#define __giftFeatureGenerator_TXX

//GIFT Includes
#include "giftFeatureGenerator.h"

//ITK includes
#include "itkDataObject.h"
#include "itkImageBase.h"

namespace gift
{

/////////////////////////////////////////////////////////
//Constructor()
template <class TImage>
FeatureGenerator<TImage>
::FeatureGenerator(){}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void
FeatureGenerator<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}


}// end namespace gift

#endif
