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

#ifndef __rtkDrawEllipsoidImageFilter_txx
#define __rtkDrawEllipsoidImageFilter_txx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkHomogeneousMatrix.h"

#include "rtkMacro.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
DrawEllipsoidImageFilter<TInputImage, TOutputImage>
::DrawEllipsoidImageFilter():
m_Axis(3,90.),
m_Center(3,0.),
m_Attenuation(1.),
m_Angle(0.)
{
}

template <class TInputImage, class TOutputImage>
void DrawEllipsoidImageFilter<TInputImage, TOutputImage>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                                                               ThreadIdType itkNotUsed(threadId) )
{
  //Getting phantom parameters
  EQPFunctionType::Pointer sqpFunctor = EQPFunctionType::New();
  FigureType ellipsoid;
  ellipsoid.semiprincipalaxis.push_back(m_Axis[0]);
  ellipsoid.semiprincipalaxis.push_back(m_Axis[1]);
  ellipsoid.semiprincipalaxis.push_back(m_Axis[2]);
  ellipsoid.center.push_back(m_Center[0]);
  ellipsoid.center.push_back(m_Center[1]);
  ellipsoid.center.push_back(m_Center[2]);
  ellipsoid.angle = m_Angle;
  ellipsoid.attenuation = m_Attenuation;

  typename TOutputImage::PointType point;
  const    TInputImage *           input = this->GetInput();

  typename itk::ImageRegionConstIterator<TInputImage> itIn( input, outputRegionForThread);
  typename itk::ImageRegionIterator<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);
  //Translate from regular expression to quadric
  sqpFunctor->Translate(ellipsoid.semiprincipalaxis);
  //Apply rotation and translation if necessary
  sqpFunctor->Rotate(ellipsoid.angle, ellipsoid.center);
  while( !itOut.IsAtEnd() )
  {
    this->GetInput()->TransformIndexToPhysicalPoint(itOut.GetIndex(), point);

    double QuadricEllip = sqpFunctor->GetA()*point[0]*point[0]   +
                          sqpFunctor->GetB()*point[1]*point[1]   +
                          sqpFunctor->GetC()*point[2]*point[2]   +
                          sqpFunctor->GetD()*point[0]*point[1]   +
                          sqpFunctor->GetE()*point[0]*point[2]   +
                          sqpFunctor->GetF()*point[1]*point[2]   +
                          sqpFunctor->GetG()*point[0] + sqpFunctor->GetH()*point[1] +
                          sqpFunctor->GetI()*point[2] + sqpFunctor->GetJ();
    if(QuadricEllip<0)
      itOut.Set( ellipsoid.attenuation + itIn.Get() );
    else
      itOut.Set( itIn.Get() );
    ++itIn;
    ++itOut;
  }
}

}// end namespace rtk

#endif
