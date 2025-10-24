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

#include "math.h"

#include "rtkIntersectionOfConvexShapes.h"

namespace rtk
{

IntersectionOfConvexShapes ::IntersectionOfConvexShapes() = default;

void
IntersectionOfConvexShapes ::SetConvexShapes(const ConvexShapeVector & _arg)
{
  m_ConvexShapes.clear();
  for (auto & convexShape : _arg)
    this->AddConvexShape(convexShape);
  this->Modified();
}

bool
IntersectionOfConvexShapes ::IsInside(const PointType & point) const
{
  for (const auto & convexShape : m_ConvexShapes)
  {
    if (!convexShape->IsInside(point))
      return false;
  }
  return true;
}

bool
IntersectionOfConvexShapes ::IsIntersectedByRay(const PointType &  rayOrigin,
                                                const VectorType & rayDirection,
                                                ScalarType &       infDist,
                                                ScalarType &       supDist) const
{
  infDist = itk::NumericTraits<ScalarType>::NonpositiveMin();
  supDist = itk::NumericTraits<ScalarType>::max();
  for (const auto & convexShape : m_ConvexShapes)
  {
    ScalarType n = NAN, f = NAN;
    if (!convexShape->IsIntersectedByRay(rayOrigin, rayDirection, n, f))
      return false;
    infDist = std::max(infDist, n);
    supDist = std::min(supDist, f);
    if (infDist >= supDist)
      return false;
  }
  return true;
}

void
IntersectionOfConvexShapes ::Rescale(const VectorType & r)
{
  Superclass::Rescale(r);
  for (auto & convexShape : m_ConvexShapes)
    convexShape->Rescale(r);
}

void
IntersectionOfConvexShapes ::Translate(const VectorType & t)
{
  Superclass::Translate(t);
  for (auto & convexShape : m_ConvexShapes)
    convexShape->Translate(t);
}

void
IntersectionOfConvexShapes ::Rotate(const RotationMatrixType & r)
{
  Superclass::Rotate(r);
  for (auto & convexShape : m_ConvexShapes)
    convexShape->Rotate(r);
}

void
IntersectionOfConvexShapes ::AddConvexShape(const ConvexShape * co)
{
  ConvexShapePointer clone = co->Clone();
  m_ConvexShapes.push_back(clone);
}

itk::LightObject::Pointer
IntersectionOfConvexShapes ::InternalClone() const
{
  LightObject::Pointer loPtr = Superclass::InternalClone();
  Self::Pointer        clone = dynamic_cast<Self *>(loPtr.GetPointer());

  clone->SetConvexShapes(this->GetConvexShapes());

  return loPtr;
}
} // end namespace rtk
