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

#ifndef __rtkWaterCalibrationImageFilter_txx
#define __rtkWaterCalibrationImageFilter_txx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkWaterCalibrationImageFilter.h"

namespace rtk
{
template <class TInputImage, class  TOutputImage>
WaterCalibrationImageFilter<TInputImage,TOutputImage>
::WaterCalibrationImageFilter()
{
  m_Order = 1.0f;
}

template <class TInputImage, class  TOutputImage>
void WaterCalibrationImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData( const OutputImageRegionType & outputRegionForThread, ThreadIdType itkNotUsed(threadId) )
{
  typename itk::ImageRegionConstIterator< TInputImage > itIn(this->GetInput(), outputRegionForThread);
  typename itk::ImageRegionIterator< TInputImage >      itOut(this->GetOutput(), outputRegionForThread);

  itIn.GoToBegin();
  itOut.GoToBegin();
  while ( !itIn.IsAtEnd() )
    {
    float v = itIn.Get();
    itOut.Set( std::pow(v, m_Order) );
    ++itIn;
    ++itOut;
    }
}
} // end namespace rtk

#endif
