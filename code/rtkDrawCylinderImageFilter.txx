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

#ifndef __rtkDrawCylinderImageFilter_txx
#define __rtkDrawCylinderImageFilter_txx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkHomogeneousMatrix.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
DrawCylinderImageFilter<TInputImage, TOutputImage>
::DrawCylinderImageFilter():
m_Axis(0.),
m_Center(0.),
m_Attenuation(1.),
m_Angle(0.)
{
}

template <class TInputImage, class TOutputImage>
void DrawCylinderImageFilter<TInputImage, TOutputImage>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                                                                               ThreadIdType itkNotUsed(threadId) )
{
  //Getting phantom parameters
  EQPFunctionType::Pointer sqpFunctor = EQPFunctionType::New();
  FigureType Cylinder;

  Cylinder.semiprincipalaxis.push_back(m_Axis[0]);
  Cylinder.semiprincipalaxis.push_back(m_Axis[1]);
  Cylinder.semiprincipalaxis.push_back(m_Axis[2]);
  Cylinder.center.push_back(m_Center[0]);
  Cylinder.center.push_back(m_Center[1]);
  Cylinder.center.push_back(m_Center[2]);
  Cylinder.angle = m_Angle;
  Cylinder.attenuation = m_Attenuation;

  typename TOutputImage::PointType point;
  const    TInputImage *           input = this->GetInput();

  typename itk::ImageRegionConstIterator<TInputImage> itIn( input, outputRegionForThread);
  typename itk::ImageRegionIterator<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);

  //Translate from regular expression to quadric
  sqpFunctor->Translate(Cylinder.semiprincipalaxis);
  //Apply rotation and translation if necessary
  sqpFunctor->Rotate(Cylinder.angle, Cylinder.center);

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
      itOut.Set(Cylinder.attenuation + itIn.Get());
    else
      itOut.Set(0.);
    ++itIn;
    ++itOut;
  }
}

}// end namespace rtk

#endif
