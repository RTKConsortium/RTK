/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkRayQuadricIntersectionImageFilter_hxx
#define rtkRayQuadricIntersectionImageFilter_hxx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkQuadricShape.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
RayQuadricIntersectionImageFilter<TInputImage, TOutputImage>::RayQuadricIntersectionImageFilter() = default;

template <class TInputImage, class TOutputImage>
ConvexShape::Pointer
RayQuadricIntersectionImageFilter<TInputImage, TOutputImage>::CreateConvexShape() const
{
  auto qo = QuadricShape::New();
  qo->SetDensity(m_Density);
  qo->SetClipPlanes(m_PlaneDirections, m_PlanePositions);
  qo->SetA(m_A);
  qo->SetB(m_B);
  qo->SetC(m_C);
  qo->SetD(m_D);
  qo->SetE(m_E);
  qo->SetF(m_F);
  qo->SetG(m_G);
  qo->SetH(m_H);
  qo->SetI(m_I);
  qo->SetJ(m_J);
  return qo.GetPointer();
}

template <class TInputImage, class TOutputImage>
void
RayQuadricIntersectionImageFilter<TInputImage, TOutputImage>::AddClipPlane(const VectorType & dir,
                                                                           const ScalarType & pos)
{
  m_PlaneDirections.push_back(dir);
  m_PlanePositions.push_back(pos);
  this->Modified();
}

} // end namespace rtk

#endif
