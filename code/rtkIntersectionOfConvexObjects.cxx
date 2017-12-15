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

#include "rtkIntersectionOfConvexObjects.h"

namespace rtk
{

IntersectionOfConvexObjects
::IntersectionOfConvexObjects()
{
}

bool
IntersectionOfConvexObjects
::IsInside(const PointType& point) const
{
  for(size_t i=0; i<m_ConvexObjects.size(); i++)
    {
    if( !m_ConvexObjects[i]->IsInside(point) )
      return false;
    }
  return true;
}

bool
IntersectionOfConvexObjects
::IsIntersectedByRay(const PointType & rayOrigin,
                     const VectorType & rayDirection,
                     ScalarType & near,
                     ScalarType & far) const
{
  near = itk::NumericTraits< ScalarType >::NonpositiveMin();
  far = itk::NumericTraits< ScalarType >::max();
  for(size_t i=0; i<m_ConvexObjects.size(); i++)
    {
    ScalarType n, f;
    if( !m_ConvexObjects[i]->IsIntersectedByRay(rayOrigin, rayDirection, n, f) )
      return false;
    near = std::max(near, n);
    far = std::min(far, f);
    if(near >= far)
      return false;
    }
  return true;
}

void
IntersectionOfConvexObjects
::Rescale(const VectorType &r)
{
  Superclass::Rescale(r);
  for(size_t i=0; i<m_ConvexObjects.size(); i++)
    m_ConvexObjects[i]->Rescale(r);
}

void
IntersectionOfConvexObjects
::Translate(const VectorType &t)
{
  Superclass::Translate(t);
  for(size_t i=0; i<m_ConvexObjects.size(); i++)
    m_ConvexObjects[i]->Translate(t);
}

void
IntersectionOfConvexObjects
::Rotate(const RotationMatrixType &r)
{
  Superclass::Rotate(r);
  for(size_t i=0; i<m_ConvexObjects.size(); i++)
    m_ConvexObjects[i]->Rotate(r);
}

void
IntersectionOfConvexObjects
::AddConvexObject(const ConvexObjectPointer &co)
{
  ConvexObjectPointer clone = co->Clone();
  m_ConvexObjects.push_back(clone);
}

itk::LightObject::Pointer
IntersectionOfConvexObjects
::InternalClone() const
{
  LightObject::Pointer loPtr = Superclass::InternalClone();
  Self::Pointer clone = dynamic_cast<Self *>(loPtr.GetPointer());

  clone->SetConvexObjects(this->GetConvexObjects());

  return loPtr;
}
} // end namespace rtk
