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

#include "rtkIntersectionOfConvexShapes.h"

namespace rtk
{

IntersectionOfConvexShapes
::IntersectionOfConvexShapes()
{
}

void
IntersectionOfConvexShapes
::SetConvexShapes(const ConvexShapeVector &_arg)
{
  if (this->m_ConvexShapes != _arg)
    {
    this->m_ConvexShapes = _arg;
    this->Modified();
    }
}

bool
IntersectionOfConvexShapes
::IsInside(const PointType& point) const
{
  for(size_t i=0; i<m_ConvexShapes.size(); i++)
    {
    if( !m_ConvexShapes[i]->IsInside(point) )
      return false;
    }
  return true;
}

bool
IntersectionOfConvexShapes
::IsIntersectedByRay(const PointType & rayOrigin,
                     const VectorType & rayDirection,
                     ScalarType & nearDist,
                     ScalarType & farDist) const
{
  nearDist = itk::NumericTraits< ScalarType >::NonpositiveMin();
  farDist = itk::NumericTraits< ScalarType >::max();
  for(size_t i=0; i<m_ConvexShapes.size(); i++)
    {
    ScalarType n, f;
    if( !m_ConvexShapes[i]->IsIntersectedByRay(rayOrigin, rayDirection, n, f) )
      return false;
    nearDist = std::max(nearDist, n);
    farDist = std::min(farDist, f);
    if(nearDist >= farDist)
      return false;
    }
  return true;
}

void
IntersectionOfConvexShapes
::Rescale(const VectorType &r)
{
  Superclass::Rescale(r);
  for(size_t i=0; i<m_ConvexShapes.size(); i++)
    m_ConvexShapes[i]->Rescale(r);
}

void
IntersectionOfConvexShapes
::Translate(const VectorType &t)
{
  Superclass::Translate(t);
  for(size_t i=0; i<m_ConvexShapes.size(); i++)
    m_ConvexShapes[i]->Translate(t);
}

void
IntersectionOfConvexShapes
::Rotate(const RotationMatrixType &r)
{
  Superclass::Rotate(r);
  for(size_t i=0; i<m_ConvexShapes.size(); i++)
    m_ConvexShapes[i]->Rotate(r);
}

void
IntersectionOfConvexShapes
::AddConvexShape(const ConvexShapePointer &co)
{
  ConvexShapePointer clone = co->Clone();
  m_ConvexShapes.push_back(clone);
}

itk::LightObject::Pointer
IntersectionOfConvexShapes
::InternalClone() const
{
  LightObject::Pointer loPtr = Superclass::InternalClone();
  Self::Pointer clone = dynamic_cast<Self *>(loPtr.GetPointer());

  clone->SetConvexShapes(this->GetConvexShapes());

  return loPtr;
}
} // end namespace rtk
