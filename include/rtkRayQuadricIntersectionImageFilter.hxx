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

#ifndef rtkRayQuadricIntersectionImageFilter_hxx
#define rtkRayQuadricIntersectionImageFilter_hxx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkRayQuadricIntersectionImageFilter.h"
#include "rtkQuadricShape.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
RayQuadricIntersectionImageFilter<TInputImage, TOutputImage>::RayQuadricIntersectionImageFilter() = default;

template <class TInputImage, class TOutputImage>
void
RayQuadricIntersectionImageFilter<TInputImage, TOutputImage>::BeforeThreadedGenerateData()
{
  if (this->GetConvexShape() == nullptr)
    this->SetConvexShape(QuadricShape::New().GetPointer());

  Superclass::BeforeThreadedGenerateData();

  auto * qo = dynamic_cast<QuadricShape *>(this->GetModifiableConvexShape());
  if (qo == nullptr)
  {
    itkExceptionMacro("This is not a QuadricShape!");
  }

  qo->SetDensity(this->GetDensity());
  qo->SetClipPlanes(this->GetPlaneDirections(), this->GetPlanePositions());
  qo->SetA(this->GetA());
  qo->SetB(this->GetB());
  qo->SetC(this->GetC());
  qo->SetD(this->GetD());
  qo->SetE(this->GetE());
  qo->SetF(this->GetF());
  qo->SetG(this->GetG());
  qo->SetH(this->GetH());
  qo->SetI(this->GetI());
  qo->SetJ(this->GetJ());
}

template <class TInputImage, class TOutputImage>
void
RayQuadricIntersectionImageFilter<TInputImage, TOutputImage>::AddClipPlane(const VectorType & dir,
                                                                           const ScalarType & pos)
{
  m_PlaneDirections.push_back(dir);
  m_PlanePositions.push_back(pos);
}

} // end namespace rtk

#endif
