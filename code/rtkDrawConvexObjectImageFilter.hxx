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

#ifndef rtkDrawConvexObjectImageFilter_hxx
#define rtkDrawConvexObjectImageFilter_hxx

#include "rtkDrawConvexObjectImageFilter.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
DrawConvexObjectImageFilter<TInputImage, TOutputImage>
::DrawConvexObjectImageFilter()
{
}

template <class TInputImage, class TOutputImage>
void
DrawConvexObjectImageFilter<TInputImage,TOutputImage>
::BeforeThreadedGenerateData()
{
  if( this->m_ConvexObject.IsNull() )
    itkExceptionMacro(<<"ConvexObject has not been set.")
}

template <class TInputImage, class TOutputImage>
void
DrawConvexObjectImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType threadId )
{
  // Local convex object since convex objects are not thread safe
  ConvexObjectPointer co = dynamic_cast<ConvexObject *>(m_ConvexObject->Clone().GetPointer());

  typename TOutputImage::PointType point;
  const    TInputImage * input = this->GetInput();

  typename itk::ImageRegionConstIterator<TInputImage> itIn( input, outputRegionForThread);
  typename itk::ImageRegionIterator<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);

  while( !itOut.IsAtEnd() )
    {
    this->GetInput()->TransformIndexToPhysicalPoint(itOut.GetIndex(), point);
    PointType p(&(point[0]));
    if(co->IsInside(p))
      itOut.Set( itIn.Get() + co->GetDensity() );
    else
      itOut.Set( itIn.Get() );
    ++itIn;
    ++itOut;
    }
}

} // end namespace rtk

#endif
