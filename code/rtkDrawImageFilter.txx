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

#ifndef __rtkCubeImageFilter_txx
#define __rtkCubeImageFilter_txx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkHomogeneousMatrix.h"
#include "rtkMacro.h"

#include "rtkDrawImageFilter.h"

namespace rtk
{

template <class TInputImage, class TOutputImage, class TSpatialObject>
DrawImageFilter<TInputImage, TOutputImage, TSpatialObject>
::DrawImageFilter()
{

}



template <class TInputImage, class TOutputImage, class TSpatialObject>
void DrawImageFilter<TInputImage, TOutputImage, TSpatialObject>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                                                              ThreadIdType itkNotUsed(threadId) )
{
  
  typename TOutputImage::PointType point;
  const    TInputImage *           input = this->GetInput();

  typename itk::ImageRegionConstIterator<TInputImage> itIn( input, outputRegionForThread);
  typename itk::ImageRegionIterator<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);
  
  std::cout << "C++::DrawImageFilter::ThreadedGenerateData " << std::endl;
  

  while( !itOut.IsAtEnd() )
  {
    this->GetInput()->TransformIndexToPhysicalPoint(itOut.GetIndex(), point);
    
    if(m_spatialObject.IsInside(point))
      itOut.Set( m_Density );
    else
      itOut.Set( itIn.Get() );
    ++itIn;
    ++itOut;
  }
  
  
}


template <class TInputImage, class TOutputImage>
myDrawCylinderImageFilter<TInputImage, TOutputImage>
::myDrawCylinderImageFilter()
{
  std::cout << "C++::myDrawCylinderImageFilter::myDrawCylinderImageFilter " << std::endl;
  this->m_spatialObject.sqpFunctor = EQPFunctionType::New();
  this->m_spatialObject.sqpFunctor->SetFigure("Cylinder");
}


// template <class TInputImage, class TOutputImage>
// void myDrawCylinderImageFilter<TInputImage, TOutputImage>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
//                                                                               ThreadIdType threadId )
// { 
//   
//   
//   std::cout << "C++::myDrawCylinderImageFilter::ThreadedGenerateData (" << threadId << ")" << std::endl;
//   DrawImageFilter<TInputImage, TOutputImage, DrawCylinderSpatialObject>::ThreadedGenerateData(outputRegionForThread, threadId);
// }




}// end namespace rtk

#endif
