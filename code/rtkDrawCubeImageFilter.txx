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

#ifndef __rtkDrawCubeImageFilter_txx
#define __rtkDrawCubeImageFilter_txx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkHomogeneousMatrix.h"

#include "rtkMacro.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
DrawCubeImageFilter<TInputImage, TOutputImage>
::DrawCubeImageFilter()
{
  m_Axis.Fill(50.);
  m_Center.Fill(0.);
  m_Density = 1.;
  m_Angle = 0.;
}

template <class TInputImage, class TOutputImage>
void DrawCubeImageFilter<TInputImage, TOutputImage>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                                                               ThreadIdType itkNotUsed(threadId) )
{
  //Getting phantom parameters
  FigureType cube;

  //Setting axis dimensions taking into account center
  // X-Axis superior
  cube.semiprincipalaxis[0] =  m_Axis[0] + m_Center[0];
  // X-Axis inferior
  cube.semiprincipalaxis[1] = -m_Axis[0] + m_Center[0];
  // Y-Axis superior
  cube.semiprincipalaxis[2] =  m_Axis[1] + m_Center[1];
  // Y-Axis inferior
  cube.semiprincipalaxis[3] = -m_Axis[1] + m_Center[1];
  // Z-Axis superior
  cube.semiprincipalaxis[4] =  m_Axis[2] + m_Center[2];
  // Z-Axis inferior
  cube.semiprincipalaxis[5] = -m_Axis[2] + m_Center[2];

  cube.angle = m_Angle;
  cube.density = m_Density;

  typename TOutputImage::PointType point;
  const    TInputImage *           input = this->GetInput();

  typename itk::ImageRegionConstIterator<TInputImage> itIn( input, outputRegionForThread);
  typename itk::ImageRegionIterator<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);

  //Set type of Figure
  //Apply rotation if necessary
  // FIXME: add rotation option
  //sqpFunctor->Rotate(cube.angle, cube.center);

  while( !itOut.IsAtEnd() )
  {
    this->GetInput()->TransformIndexToPhysicalPoint(itOut.GetIndex(), point);

    if(point[0]<cube.semiprincipalaxis[0] && point[0]>cube.semiprincipalaxis[1] &&
       point[1]<cube.semiprincipalaxis[2] && point[1]>cube.semiprincipalaxis[3] &&
       point[2]<cube.semiprincipalaxis[4] && point[2]>cube.semiprincipalaxis[5])
      itOut.Set( cube.density + itIn.Get() );
    else
      itOut.Set( itIn.Get() );
    ++itIn;
    ++itOut;
  }
}

}// end namespace rtk

#endif
