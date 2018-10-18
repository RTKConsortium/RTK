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

#ifndef rtkRayBoxIntersectionImageFilter_hxx
#define rtkRayBoxIntersectionImageFilter_hxx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkBoxShape.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
RayBoxIntersectionImageFilter<TInputImage, TOutputImage>
::RayBoxIntersectionImageFilter():
  m_Density(1.),
  m_BoxMin(0.),
  m_BoxMax(0.)
{
  m_Direction.SetIdentity();
}

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  if( this->GetConvexShape() == ITK_NULLPTR )
    this->SetConvexShape( BoxShape::New().GetPointer() );

  Superclass::BeforeThreadedGenerateData();

  BoxShape * qo = dynamic_cast< BoxShape * >( this->GetModifiableConvexShape() );
  if( qo == ITK_NULLPTR )
    {
    itkExceptionMacro("This is not a BoxShape!");
    }

  qo->SetDensity( this->GetDensity() );
  qo->SetClipPlanes( this->GetPlaneDirections(), this->GetPlanePositions() );
  qo->SetBoxMin(this->GetBoxMin());
  qo->SetBoxMax(this->GetBoxMax());
}

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage, TOutputImage>
::AddClipPlane(const VectorType & dir, const ScalarType & pos)
{
  m_PlaneDirections.push_back(dir);
  m_PlanePositions.push_back(pos);
}

template <class TInputImage, class TOutputImage>
void
RayBoxIntersectionImageFilter<TInputImage,TOutputImage>
::SetBoxFromImage(const ImageBaseType *_arg, bool /*bWithExternalHalfPixelBorder*/ )
{
  if( this->GetConvexShape() == ITK_NULLPTR )
    this->SetConvexShape( BoxShape::New().GetPointer() );
  BoxShape * qo = dynamic_cast< BoxShape * >( this->GetModifiableConvexShape() );
  if( qo == ITK_NULLPTR )
    {
    itkExceptionMacro("This is not a BoxShape!");
    }
  qo->SetBoxFromImage(_arg);
  SetBoxMin(qo->GetBoxMin());
  SetBoxMax(qo->GetBoxMax());
  SetDirection(qo->GetDirection());
}

}// end namespace rtk

#endif
