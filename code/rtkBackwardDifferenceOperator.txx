/*=========================================================================
 *
 *  Copyright Insight Software Consortium
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
#ifndef __rtkBackwardDifferenceOperator_hxx
#define __rtkBackwardDifferenceOperator_hxx
#include "rtkBackwardDifferenceOperator.h"

#include "itkNumericTraits.h"

namespace rtk
{
  using namespace itk;

template< typename TPixel, unsigned int VDimension, typename TAllocator >
typename BackwardDifferenceOperator< TPixel, VDimension, TAllocator >
::CoefficientVector
BackwardDifferenceOperator< TPixel, VDimension, TAllocator >
::GenerateCoefficients()
{
  const unsigned int w = 3;
  CoefficientVector  coeff(w);

  coeff[0] = -1;
  coeff[1] = 1;
  coeff[2] = 0;

  return coeff;
}
} // end of namespace rtk and itk

#endif
