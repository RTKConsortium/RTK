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

#ifndef rtkSumOfSquaresImageFilter_hxx
#define rtkSumOfSquaresImageFilter_hxx

#include "rtkSumOfSquaresImageFilter.h"
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

namespace rtk
{

template <class TOutputImage>
SumOfSquaresImageFilter<TOutputImage>::SumOfSquaresImageFilter()
{
  m_SumOfSquares = 0;
  m_VectorOfPartialSSs.clear();
  this->SetDynamicMultiThreading(false);
}

template <class TOutputImage>
void
SumOfSquaresImageFilter<TOutputImage>::BeforeThreadedGenerateData()
{
  m_VectorOfPartialSSs.clear();
  for (unsigned int thread = 0; thread < this->GetNumberOfWorkUnits(); thread++)
    m_VectorOfPartialSSs.push_back(0);
}

template <class TOutputImage>
void
SumOfSquaresImageFilter<TOutputImage>::ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread,
                                                            itk::ThreadIdType             threadId)
{
  itk::ImageRegionConstIterator<TOutputImage> inIt(this->GetInput(), outputRegionForThread);
  itk::ImageRegionIterator<TOutputImage>      outIt(this->GetOutput(), outputRegionForThread);

  while (!outIt.IsAtEnd())
  {
    m_VectorOfPartialSSs[threadId] += inIt.Get() * inIt.Get();

    // Pass the first input through unmodified
    outIt.Set(inIt.Get());

    // Move iterators
    ++inIt;
    ++outIt;
  }
}

template <class TOutputImage>
void
SumOfSquaresImageFilter<TOutputImage>::AfterThreadedGenerateData()
{
  m_SumOfSquares = 0;
  for (unsigned int thread = 0; thread < this->GetNumberOfWorkUnits(); thread++)
    m_SumOfSquares += m_VectorOfPartialSSs[thread];
}

} // end namespace rtk

#endif
