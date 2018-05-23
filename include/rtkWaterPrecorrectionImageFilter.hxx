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

#ifndef rtkWaterPrecorrectionImageFilter_hxx
#define rtkWaterPrecorrectionImageFilter_hxx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkWaterPrecorrectionImageFilter.h"

namespace rtk
{
template <class TInputImage, class  TOutputImage>
WaterPrecorrectionImageFilter<TInputImage, TOutputImage>
::WaterPrecorrectionImageFilter()
{
  m_Coefficients.push_back(0.0);  // No correction by default
  m_Coefficients.push_back(1.0);
}

template <class TInputImage, class  TOutputImage>
void WaterPrecorrectionImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType itkNotUsed(threadId) )
{
  const int csize = m_Coefficients.size();

  typename itk::ImageRegionConstIterator< TInputImage > itIn(this->GetInput(), outputRegionForThread);
  typename itk::ImageRegionIterator< TOutputImage >     itOut(this->GetOutput(), outputRegionForThread);

  if ( csize >= 3 )
    {
    itIn.GoToBegin();
    itOut.GoToBegin();

    while ( !itIn.IsAtEnd() )
      {
      float v = itIn.Get();
      float out = m_Coefficients[0] + m_Coefficients[1] * v;
      float bpow = v * v;

      for ( int i = 2; i < csize; i++ )
        {
        out += m_Coefficients[i] * bpow;
        bpow = bpow * v;
        }
      itOut.Set(out);

      ++itIn;
      ++itOut;
      }
    }
  else if ( ( csize == 2 ) && ( ( m_Coefficients[0] != 0 ) || ( m_Coefficients[1] != 1 ) ) )
    {
    itIn.GoToBegin();
    itOut.GoToBegin();
    while ( !itIn.IsAtEnd() )
      {
      itOut.Set( m_Coefficients[0] + m_Coefficients[1] * itIn.Get() );
      ++itIn;
      ++itOut;
      }
    }
  else if ( ( csize == 1 ) && ( m_Coefficients[0] != 0 ) )
    {
    itIn.GoToBegin();
    itOut.GoToBegin();
    while ( !itIn.IsAtEnd() )
      {
      itOut.Set( m_Coefficients[0]);
      ++itIn;
      ++itOut;
      }
    }
}
} // end namespace rtk

#endif // rtkWaterPrecorrectionImageFilter_hxx
