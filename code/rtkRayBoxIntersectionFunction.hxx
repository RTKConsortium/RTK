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
}

template < class TCoordRep, unsigned int VBoxDimension >
bool
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::Evaluate( const VectorType& rayDirection )
{
  // http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
  // BI <-> m_BoxMin
  // Bh <-> m_BoxMax
  // Ro <-> m_RayOrigin
  // Rd <-> rayDirection
  // Tnear <-> m_NearestDistance
  // Tfar <-> m_FarthestDistance
  m_RayDirection = rayDirection;
  m_NearestDistance = itk::NumericTraits< TCoordRep >::NonpositiveMin();
  m_FarthestDistance = itk::NumericTraits< TCoordRep >::max();
  TCoordRep T1, T2, invRayDir;
  for(unsigned int i=0; i<VBoxDimension; i++)
    {
    if(rayDirection[i] == itk::NumericTraits< TCoordRep >::ZeroValue())
      if(m_RayOrigin[i]<m_BoxMin[i] || m_RayOrigin[i]>m_BoxMax[i])
        return false;

    invRayDir = 1/rayDirection[i];
    T1 = (m_BoxMin[i] - m_RayOrigin[i]) * invRayDir;
    T2 = (m_BoxMax[i] - m_RayOrigin[i]) * invRayDir;
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
::SetBoxFromImage( ImageBaseConstPointer img )
{
  if(VBoxDimension != img->GetImageDimension())
    itkGenericExceptionMacro(<< "Box and image dimensions must agree");

  // Box corner 1
  m_BoxMin = img->GetOrigin().GetVectorFromOrigin();
  m_BoxMin -= img->GetSpacing() * 0.5;

  // Box corner 2
  m_BoxMax = m_BoxMin;
  for(unsigned int i=0; i<VBoxDimension; i++)
    m_BoxMax[i] += img->GetSpacing()[i] * img->GetLargestPossibleRegion().GetSize()[i];

  // Sort
  for(unsigned int i=0; i<VBoxDimension; i++)
    if(m_BoxMin[i]>m_BoxMax[i])
      std::swap( m_BoxMin[i], m_BoxMax[i] );
}

} // namespace rtk

#endif // rtkRayBoxIntersectionFunction_hxx
