#ifndef __rtkConjugateGradientOperator_txx
#define __rtkConjugateGradientOperator_txx

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
