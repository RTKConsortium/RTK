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

#include "rtkBoxShape.h"

namespace rtk
{

BoxShape ::BoxShape()
  : m_BoxMin(0.)
  , m_BoxMax(0.)
{
  m_Direction.SetIdentity();
}

bool
BoxShape ::IsInside(const PointType & point) const
{
  RotationMatrixType dirt;
  dirt = m_Direction.GetTranspose();
  PointType t = dirt * point;
  PointType min = dirt * m_BoxMin;
  PointType max = dirt * m_BoxMax;
  if (t[0] < min[0] || t[0] > max[0] || t[1] < min[1] || t[1] > max[1] || t[2] < min[2] || t[2] > max[2])
    return false;
  return ApplyClipPlanes(point);
}

bool
BoxShape ::IsIntersectedByRay(const PointType &  rayOrigin,
                              const VectorType & rayDirection,
                              double &           nearDist,
                              double &           farDist) const
{
  // Apply transform
  RotationMatrixType dirt;
  dirt = m_Direction.GetTranspose();
  PointType  org = dirt * rayOrigin;
  VectorType dir = dirt * rayDirection;
  PointType  min = dirt * m_BoxMin;
  PointType  max = dirt * m_BoxMax;
  for (unsigned int i = 0; i < Dimension; i++)
    if (min[i] > max[i])
      std::swap(min[i], max[i]);

  // To account for m_Direction, everything (ray source and direction + boxmin/boxmax)
  // is rotated with its inverse, m_DirectionT. Then, the box is aligned with the
  // axes of the coordinate system and the algorithm at this hyperlink used:
  // https://education.siggraph.org/static/HyperGraph/raytrace/rtinter3.htm
  // Note that the variables at this page have been renamed:
  // BI <-> min
  // Bh <-> max
  // Ro <-> org
  // Rd <-> dir
  // Tnear <-> nearDist
  // Tfar <-> farDist
  nearDist = itk::NumericTraits<ScalarType>::NonpositiveMin();
  farDist = itk::NumericTraits<ScalarType>::max();
  ScalarType T1 = NAN, T2 = NAN, invRayDir = NAN;
  for (unsigned int i = 0; i < Dimension; i++)
  {
    if (dir[i] == itk::NumericTraits<ScalarType>::ZeroValue() && (org[i] < min[i] || org[i] > max[i]))
      return false;

    invRayDir = 1 / dir[i];
    T1 = (min[i] - org[i]) * invRayDir;
    T2 = (max[i] - org[i]) * invRayDir;
    if (T1 > T2)
      std::swap(T1, T2);
    if (T1 > nearDist)
      nearDist = T1;
    if (T2 < farDist)
      farDist = T2;
    if (nearDist > farDist)
      return false;
    if (farDist < itk::NumericTraits<ScalarType>::ZeroValue())
      return false;
  }

  return ApplyClipPlanes(rayOrigin, rayDirection, nearDist, farDist);
}

void
BoxShape ::Rescale(const VectorType & r)
{
  Superclass::Rescale(r);
  for (unsigned int i = 0; i < Dimension; i++)
  {
    m_BoxMin[i] *= r[i];
    m_BoxMax[i] *= r[i];
  }
}

void
BoxShape ::Translate(const VectorType & t)
{
  Superclass::Translate(t);
  m_BoxMin += t;
  m_BoxMax += t;
}

void
BoxShape ::Rotate(const RotationMatrixType & r)
{
  Superclass::Rotate(r);
  m_Direction = m_Direction * r;
  m_BoxMin = m_Direction * m_BoxMin;
  m_BoxMax = m_Direction * m_BoxMax;
}

itk::LightObject::Pointer
BoxShape ::InternalClone() const
{
  LightObject::Pointer loPtr = Superclass::InternalClone();
  Self::Pointer        clone = dynamic_cast<Self *>(loPtr.GetPointer());

  clone->SetBoxMin(this->GetBoxMin());
  clone->SetBoxMax(this->GetBoxMax());
  clone->SetDirection(this->GetDirection());

  return loPtr;
}

void
BoxShape ::SetBoxFromImage(const ImageBaseType * img, bool bWithExternalHalfPixelBorder)
{
  if (Dimension != img->GetImageDimension())
    itkGenericExceptionMacro(<< "BoxShape and image dimensions must agree");

  // BoxShape corner 1
  m_BoxMin = img->GetOrigin();
  if (bWithExternalHalfPixelBorder)
    m_BoxMin -= img->GetDirection() * img->GetSpacing() * 0.5;

  // BoxShape corner 2
  VectorType max;
  for (unsigned int i = 0; i < Dimension; i++)
    if (bWithExternalHalfPixelBorder)
      max[i] = img->GetSpacing()[i] * img->GetLargestPossibleRegion().GetSize()[i];
    else
      max[i] = img->GetSpacing()[i] * (img->GetLargestPossibleRegion().GetSize()[i] - 1);
  max = img->GetDirection() * max;
  m_BoxMax = m_BoxMin + max;

  SetDirection(img->GetDirection());
}

} // end namespace rtk
