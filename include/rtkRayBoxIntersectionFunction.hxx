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

#ifndef rtkRayBoxIntersectionFunction_hxx
#define rtkRayBoxIntersectionFunction_hxx

#include "rtkRayBoxIntersectionFunction.h"

namespace rtk
{

template < class TCoordRep, unsigned int VBoxDimension >
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::RayBoxIntersectionFunction():
  m_BoxMin(0.),
  m_BoxMax(0.),
  m_RayOrigin(0.),
  m_RayDirection(0.),
  m_NearestDistance(0.),
  m_FarthestDistance(0.)
{
  this->m_Direction.SetIdentity();
  this->m_DirectionT.SetIdentity();
}

template < class TCoordRep, unsigned int VBoxDimension >
bool
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::Evaluate( const VectorType& rayDirection )
{
  // To account for m_Direction, everything (ray source and direction + boxmin/boxmax)
  // is rotated with its inverse, m_DirectionT. Then, the box is aligned with the
  // axes of the coordinate system and the algorithm at this hyperlink used:
  // http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
  // Note that the variables at this page have been renamed:
  // BI <-> m_BoxMinT
  // Bh <-> m_BoxMaxT
  // Ro <-> m_RayOriginT
  // Rd <-> rayDirectionT
  // Tnear <-> m_NearestDistance
  // Tfar <-> m_FarthestDistance
  m_RayDirection = rayDirection;
  VectorType rayDirectionT = this->m_DirectionT * rayDirection;
  VectorType rayOriginT = this->m_DirectionT * this->m_RayOrigin;
  m_NearestDistance = itk::NumericTraits< TCoordRep >::NonpositiveMin();
  m_FarthestDistance = itk::NumericTraits< TCoordRep >::max();
  TCoordRep T1, T2, invRayDir;
  for(unsigned int i=0; i<VBoxDimension; i++)
    {
    if(rayDirectionT[i] == itk::NumericTraits< TCoordRep >::ZeroValue() &&
       (rayOriginT[i]<m_BoxMinT[i] || rayOriginT[i]>m_BoxMaxT[i]) )
      return false;

    invRayDir = 1/rayDirectionT[i];
    T1 = (m_BoxMinT[i] - rayOriginT[i]) * invRayDir;
    T2 = (m_BoxMaxT[i] - rayOriginT[i]) * invRayDir;
    if(T1>T2) std::swap( T1, T2 );
    if(T1>m_NearestDistance) m_NearestDistance = T1;
    if(T2<m_FarthestDistance) m_FarthestDistance = T2;
    if(m_NearestDistance>m_FarthestDistance) return false;
    if(m_FarthestDistance<0) return false;
    }
  return true;
}

template < class TCoordRep, unsigned int VBoxDimension >
void
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::SetBoxFromImage( const ImageBaseType *img, bool bWithExternalHalfPixelBorder )
{
  if(VBoxDimension != img->GetImageDimension())
    itkGenericExceptionMacro(<< "Box and image dimensions must agree");

  // Box corner 1
  m_BoxMin = img->GetOrigin().GetVectorFromOrigin();
  if(bWithExternalHalfPixelBorder)
    m_BoxMin -= img->GetDirection() * img->GetSpacing() * 0.5;

  // Box corner 2
  m_BoxMax = m_BoxMin;
  for(unsigned int i=0; i<VBoxDimension; i++)
    if(bWithExternalHalfPixelBorder)
      m_BoxMax[i] = img->GetSpacing()[i] * img->GetLargestPossibleRegion().GetSize()[i];
    else
      m_BoxMax[i] = img->GetSpacing()[i] * (img->GetLargestPossibleRegion().GetSize()[i]-1);
  m_BoxMax = m_BoxMin + img->GetDirection() * m_BoxMax;

  this->SetDirection( img->GetDirection() );
}

template < class TCoordRep, unsigned int VBoxDimension >
typename RayBoxIntersectionFunction<TCoordRep, VBoxDimension>::VectorType
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::GetBoxMin()
{
  return this->m_BoxMin;
}

template < class TCoordRep, unsigned int VBoxDimension >
void
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::SetBoxMin(const VectorType _arg)
{
  m_BoxMin = _arg;
  TransformBoxCornersWithDirection();
}

template < class TCoordRep, unsigned int VBoxDimension >
typename RayBoxIntersectionFunction<TCoordRep, VBoxDimension>::VectorType
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::GetBoxMax()
{
  return this->m_BoxMax;
}

template < class TCoordRep, unsigned int VBoxDimension >
void
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::SetBoxMax(const VectorType _arg)
{
  m_BoxMax = _arg;
  TransformBoxCornersWithDirection();
}

template < class TCoordRep, unsigned int VBoxDimension >
typename RayBoxIntersectionFunction<TCoordRep, VBoxDimension>::DirectionType
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::GetDirection()
{
  return this->m_Direction;
}

template < class TCoordRep, unsigned int VBoxDimension >
void
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::SetDirection(const DirectionType _arg)
{
  this->m_Direction = _arg;
  this->m_DirectionT = this->m_Direction.GetTranspose();
  TransformBoxCornersWithDirection();
}

template < class TCoordRep, unsigned int VBoxDimension >
void
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::TransformBoxCornersWithDirection( )
{
  this->m_BoxMinT = this->m_DirectionT * this->m_BoxMin;
  this->m_BoxMaxT = this->m_DirectionT * this->m_BoxMax;

  // Sort
  for(unsigned int i=0; i<VBoxDimension; i++)
    if(m_BoxMinT[i]>m_BoxMaxT[i])
      std::swap( m_BoxMinT[i], m_BoxMaxT[i] );
}

} // namespace rtk

#endif // rtkRayBoxIntersectionFunction_hxx
