/*=========================================================================

  Program:  GIFT Weight Combiner
  Module:   giftWeightCombiner.txx
  Language: C++
  Date:     2005/11/23
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftWeightCombiner_TXX
#define __giftWeightCombiner_TXX

//GIFT Includes
#include "giftWeightCombiner.h"

//ITK includes

namespace gift
{

/////////////////////////////////////////////////////////
//Constructor()
template <class TImage>
WeightCombiner<TImage>
::WeightCombiner()
{

}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage>
void
WeightCombiner<TImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}


}// end namespace gift

#endif
