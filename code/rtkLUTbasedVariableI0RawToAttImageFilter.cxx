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

#ifndef __rtkLUTbasedVarI0RawToAttImageFilter_h

#include "rtkLUTbasedVariableI0RawToAttImageFilter.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

namespace rtk
{

LUTbasedVariableI0RawToAttImageFilter::LUTbasedVariableI0RawToAttImageFilter()
{
  m_LnILUT[0] = 0;
  for (int i = 1; i < lutSize; ++i) {
    m_LnILUT[i] = -log(float(i));
  }

  m_I0 = lutSize - 1;
  m_lnI0 = log(float(m_I0));
}

void LUTbasedVariableI0RawToAttImageFilter::BeforeThreadedGenerateData()
{
  float i0 = float(m_I0);
  m_lnI0 = log((i0 > 1) ? i0 : 1.0f);
}

void LUTbasedVariableI0RawToAttImageFilter
::ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId)
{
  itk::ImageRegionConstIterator<InputImageType>  itIn(this->GetInput(), outputRegionForThread);
  itk::ImageRegionIterator<OutputImageType>      itOut(this->GetOutput(), outputRegionForThread);

  itIn.GoToBegin();
  itOut.GoToBegin();
  while (!itIn.IsAtEnd())
  {
    float value = m_lnI0 + m_LnILUT[itIn.Get()];
    itOut.Set( (value>=0)?value:0.0f );
    ++itIn;
    ++itOut;
  }
}

} // end namespace rtk

#endif // __rtkLUTbasedVariableI0RawToAttImageFilter_cxx
