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

#ifndef rtkRayBoxIntersectionImageFilter_hxx
#define rtkRayBoxIntersectionImageFilter_hxx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkBoxShape.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
RayBoxIntersectionImageFilter<TInputImage, TOutputImage>::RayBoxIntersectionImageFilter()
{
  m_Direction.SetIdentity();
}

template <class TInputImage, class TOutputImage>
ConvexShape::Pointer
RayBoxIntersectionImageFilter<TInputImage, TOutputImage>::CreateConvexShape() const
{
  auto qo = BoxShape::New();
  qo->SetDensity(m_Density);
  qo->SetClipPlanes(m_PlaneDirections, m_PlanePositions);
  qo->SetBoxMin(m_BoxMin);
  qo->SetBoxMax(m_BoxMax);
  qo->SetDirection(m_Direction);
  return qo.GetPointer();
}

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage, TOutputImage>::AddClipPlane(const VectorType & dir, const ScalarType & pos)
{
  m_PlaneDirections.push_back(dir);
  m_PlanePositions.push_back(pos);
  this->Modified();
}

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage, TOutputImage>::SetBoxFromImage(const ImageBaseType * _arg,
                                                                          bool bWithExternalHalfPixelBorder)
{
  auto box = BoxShape::New();
  box->SetBoxFromImage(_arg, bWithExternalHalfPixelBorder);
  this->SetBoxMin(box->GetBoxMin());
  this->SetBoxMax(box->GetBoxMax());
  this->SetDirection(box->GetDirection());
}

} // end namespace rtk

#endif
