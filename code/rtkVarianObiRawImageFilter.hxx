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

#ifndef rtkVarianObiRawImageFilter_hxx
#define rtkVarianObiRawImageFilter_hxx

#include "rtkVarianObiRawImageFilter.h"
#include "rtkI0EstimationProjectionFilter.h"

namespace rtk
{

template< typename TInputImage, typename TOutputImage >
VarianObiRawImageFilter< TInputImage, TOutputImage >
::VarianObiRawImageFilter():
  m_I0(139000.),
  m_IDark(0.)
{
}

template< typename TInputImage, typename TOutputImage >
void
VarianObiRawImageFilter< TInputImage, TOutputImage >
::BeforeThreadedGenerateData()
{
  typedef rtk::I0EstimationProjectionFilter<TInputImage> I0EstimationType;
  I0EstimationType * i0est = dynamic_cast<I0EstimationType*>( this->GetInput()->GetSource().GetPointer() );
  if(i0est)
    {
    m_I0 = (double)i0est->GetI0();
    }
  this->GetFunctor().SetI0(m_I0);
  this->GetFunctor().SetIDark(m_IDark);
}

} // end namespace rtk

#endif
