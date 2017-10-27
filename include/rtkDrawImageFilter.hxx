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

#ifndef rtkDrawImageFilter_hxx
#define rtkDrawImageFilter_hxx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkHomogeneousMatrix.h"
#include "rtkMacro.h"

#include "rtkDrawImageFilter.h"

namespace rtk
{

template <class TInputImage, class TOutputImage, class TSpatialObject, typename TFunction>
DrawImageFilter<TInputImage, TOutputImage, TSpatialObject, TFunction>
::DrawImageFilter()
{
  m_Density = 1.; //backward compatibility
}

template <class TInputImage, class TOutputImage, class TSpatialObject, typename TFunction>
void DrawImageFilter<TInputImage, TOutputImage, TSpatialObject, TFunction>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                                                              ThreadIdType itkNotUsed(threadId) )
{
  typename TOutputImage::PointType point;
  const    TInputImage *           input = this->GetInput();

  typename itk::ImageRegionConstIterator<TInputImage> itIn( input, outputRegionForThread);
  typename itk::ImageRegionIterator<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);


  while( !itOut.IsAtEnd() )
  {
    this->GetInput()->TransformIndexToPhysicalPoint(itOut.GetIndex(), point);

    if(m_SpatialObject.IsInside(point))
      itOut.Set( m_Fillerfunctor( m_Density, itIn.Get() ));
    else
      itOut.Set( itIn.Get() );
    ++itIn;
    ++itOut;
  }


}

}// end namespace rtk

#endif
