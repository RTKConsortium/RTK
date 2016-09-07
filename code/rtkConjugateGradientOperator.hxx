/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkConjugateGradientOperator_hxx
#define rtkConjugateGradientOperator_hxx

#include "rtkConjugateGradientOperator.h"

namespace rtk
{

template<typename OutputImageType>
ConjugateGradientOperator<OutputImageType>::ConjugateGradientOperator(){}

template< typename OutputImageType>
void
ConjugateGradientOperator<OutputImageType>
::SetX(const OutputImageType* OutputImage)
{
  this->SetNthInput(0, const_cast<OutputImageType*>(OutputImage));
}


}// end namespace


#endif
