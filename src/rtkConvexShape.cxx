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

#include "rtkConvexShape.h"

namespace rtk
{

ConvexShape
::ConvexShape():
    m_Density(0.)
{
}

bool
ConvexShape
::IsInside(const PointType& /*point*/) const
{
  itkExceptionMacro(<< "This method should have been reimplemented in base classe");
  return false;
}

bool
ConvexShape
::IsIntersectedByRay(const PointType & /*rayOrigin*/,
                     const VectorType & /*rayDirection*/,
                     ScalarType & /*nearDist*/,
                     ScalarType & /*farDist*/) const
{
  itkExceptionMacro(<< "This method should have been reimplemented in base classe");
  return false;
}

void
ConvexShape
::Rescale(const VectorType &r)
{
  for(size_t i=0; i<m_PlaneDirections.size(); i++)
    {
    for(unsigned int j=0; j<Dimension; j++)
      {
      m_PlaneDirections[i][j] /= r[j];
      }
    }
}

void
ConvexShape
::Translate(const VectorType &t)
{
  for(size_t i=0; i<m_PlaneDirections.size(); i++)
    {
    m_PlanePositions[i] += m_PlaneDirections[i] * t;
    }
}

void
ConvexShape
::Rotate(const RotationMatrixType &r)
{
  for(size_t i=0; i<m_PlaneDirections.size(); i++)
    {
    m_PlaneDirections[i] = r * m_PlaneDirections[i];
    }
}

itk::LightObject::Pointer
ConvexShape
::InternalClone() const
{
  LightObject::Pointer loPtr = Superclass::InternalClone();
  Self::Pointer clone = dynamic_cast<Self *>(loPtr.GetPointer());

  clone->SetDensity( this->GetDensity() );
  clone->SetClipPlanes( this->GetPlaneDirections(), this->GetPlanePositions() );

  return loPtr;
}

void
ConvexShape
::AddClipPlane(const VectorType & dir, const ScalarType & pos)
{
  for(size_t i=0; i<m_PlaneDirections.size(); i++)
    {
    if(dir==m_PlaneDirections[i] && pos==m_PlanePositions[i])
      return;
    }
  m_PlaneDirections.push_back(dir);
  m_PlanePositions.push_back(pos);
}

void
ConvexShape
::SetClipPlanes(const std::vector<VectorType> & dir, const std::vector<ScalarType> & pos)
{
  m_PlaneDirections = dir;
  m_PlanePositions = pos;
}

bool
ConvexShape
::ApplyClipPlanes(const PointType & rayOrigin,
                  const VectorType & rayDirection,
                  double & nearDist,
                  double & farDist) const
{
  for(size_t i=0; i<m_PlaneDirections.size(); i++)
    {
    ScalarType rayDirPlaneDir = rayDirection * m_PlaneDirections[i];

    // Ray parallel to plane
    if( rayDirPlaneDir == itk::NumericTraits<ScalarType>::ZeroValue() )
      {
      if(rayOrigin * m_PlaneDirections[i] < m_PlanePositions[i])
        continue;
      else
        return false;
      }

    // Compute plane distance in ray direction
    ScalarType planeDist = (m_PlanePositions[i] - rayOrigin * m_PlaneDirections[i]) / rayDirPlaneDir;

    // If plane is pointing in the same direction as the ray
    if( rayDirPlaneDir >= 0 )
      {
      if(planeDist<=nearDist)
        return false;
      farDist = std::min(farDist, planeDist);
      }
    else
      {
      if(planeDist>=farDist)
        return false;
      nearDist = std::max(nearDist, planeDist);
      }
    }
  return true;
}

bool
ConvexShape
::ApplyClipPlanes(const PointType & point) const
{
  for(size_t i=0; i<m_PlaneDirections.size(); i++)
    {
    if(point*m_PlaneDirections[i] >= m_PlanePositions[i])
      return false;
    }
  return true;
}

}
