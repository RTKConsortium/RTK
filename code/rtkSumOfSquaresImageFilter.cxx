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

#include "rtkSumOfSquaresImageFilter.h"

namespace rtk
{

template <>
void
SumOfSquaresImageFilter<itk::VectorImage<float, 3>>
::ThreadedGenerateData(const itk::VectorImage<float, 3>::RegionType& outputRegionForThread, itk::ThreadIdType threadId)
{
  itk::ImageRegionConstIterator<itk::VectorImage<float, 3>> inIt(this->GetInput(), outputRegionForThread);
  itk::ImageRegionIterator<itk::VectorImage<float, 3>> outIt(this->GetOutput(), outputRegionForThread);

  while(!outIt.IsAtEnd())
    {
    for (unsigned int component=0; component<this->GetInput()->GetVectorLength(); component++)
      {
      m_VectorOfPartialSSs[threadId] += inIt.Get()[component] * inIt.Get()[component];
      }

    // Pass the first input through unmodified
    outIt.Set(inIt.Get());

    // Move iterators
    ++inIt;
    ++outIt;
    }
}

} // end namespace rtk
