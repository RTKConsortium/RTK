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

#ifndef rtkRayEllipsoidIntersectionImageFilter_hxx
#define rtkRayEllipsoidIntersectionImageFilter_hxx


#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkQuadric.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>
::RayEllipsoidIntersectionImageFilter():
    m_Density(1.),
    m_Angle(0.)
{
  m_Center.Fill(0.);
  m_Axis.Fill(0.);
}

template <class TInputImage, class TOutputImage>
void
RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  if( this->GetConvexObject() == ITK_NULLPTR )
    this->SetConvexObject( Quadric::New().GetPointer() );
  Superclass::BeforeThreadedGenerateData();
  Quadric * qo = dynamic_cast< Quadric * >( this->GetConvexObject() );
  if( qo == ITK_NULLPTR )
    {
    itkExceptionMacro("This is not a Quadric!");
    }
  qo->SetEllipsoid(m_Center, m_Axis, m_Angle);
  qo->SetDensity(m_Density);
  qo->SetClippingPlanes( this->GetPlaneDirections(), this->GetPlanePositions() );
}

template <class TInputImage, class TOutputImage>
void
RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>
::AddClippingPlane(const VectorType & dir, const ScalarType & pos)
{
  m_PlaneDirections.push_back(dir);
  m_PlanePositions.push_back(pos);
}

}// end namespace rtk

#endif
