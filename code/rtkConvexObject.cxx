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

#include "rtkConvexObject.h"

namespace rtk
{
ConvexObject
::ConvexObject():
    m_Density(0.)
{
}
bool
ConvexObject
::IsInside(const PointType& point) const
{
  itkExceptionMacro(<< "This method should have been reimplemented in base classe");
  return false;
}

bool
ConvexObject
::IsIntersectedByRay(const PointType & rayOrigin,
                     const VectorType & rayDirection,
                     ScalarType & near,
                     ScalarType & far) const
{
  itkExceptionMacro(<< "This method should have been reimplemented in base classe");
  return false;
}

void
ConvexObject
::Rescale(const VectorType &r)
{
  for(size_t i=0; i<m_PlaneDirections.size(); i++)
    {
    for(int j=0; j<Dimension; j++)
      {
      m_PlaneDirections[i][j] /= r[j];
      }
    }
}

void
ConvexObject
::Translate(const VectorType &t)
{
  for(size_t i=0; i<m_PlaneDirections.size(); i++)
    {
    m_PlanePositions[i] += m_PlaneDirections[i] * t;
    }
}

void
ConvexObject
::Rotate(const RotationMatrixType &r)
{
  for(size_t i=0; i<m_PlaneDirections.size(); i++)
    {
    m_PlaneDirections[i] = r * m_PlaneDirections[i];
    }
}

itk::LightObject::Pointer
ConvexObject
::InternalClone() const
{
  LightObject::Pointer loPtr = Superclass::InternalClone();
  Self::Pointer clone = dynamic_cast<Self *>(loPtr.GetPointer());

  clone->SetDensity( this->GetDensity() );
  clone->SetClippingPlanes( this->GetPlaneDirections(), this->GetPlanePositions() );

  return loPtr;
}

void
ConvexObject
::AddClippingPlane(const VectorType & dir, const ScalarType & pos)
{
  m_PlaneDirections.push_back(dir);
  m_PlanePositions.push_back(pos);
}

void
ConvexObject
::SetClippingPlanes(const std::vector<VectorType> & dir, const std::vector<ScalarType> & pos)
{
  m_PlaneDirections = dir;
  m_PlanePositions = pos;
}

bool
ConvexObject
::ApplyClippingPlanes(const PointType & rayOrigin,
                      const VectorType & rayDirection,
                      double & near,
                      double & far) const
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
      if(planeDist<=near)
        return false;
      far = std::min(far, planeDist);
      }
    else
      {
      if(planeDist>=far)
        return false;
      near = std::max(near, planeDist);
      }
    }
  return true;
}

bool
ConvexObject
::ApplyClippingPlanes(const PointType & point) const
{
  for(size_t i=0; i<m_PlaneDirections.size(); i++)
    {
    if(point*m_PlaneDirections[i] >= m_PlanePositions[i])
      return false;
    }
  return true;
}

}
