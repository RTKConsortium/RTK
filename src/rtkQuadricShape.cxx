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

#include "rtkQuadricShape.h"

namespace rtk
{

QuadricShape ::QuadricShape() = default;

bool
QuadricShape ::IsInside(const PointType & point) const
{
  return (IsInsideQuadric(point) && ApplyClipPlanes(point));
}

bool
QuadricShape ::IsInsideQuadric(const PointType & point) const
{
  ScalarType QuadricEllip = this->GetA() * point[0] * point[0] + this->GetB() * point[1] * point[1] +
                            this->GetC() * point[2] * point[2] + this->GetD() * point[0] * point[1] +
                            this->GetE() * point[0] * point[2] + this->GetF() * point[1] * point[2] +
                            this->GetG() * point[0] + this->GetH() * point[1] + this->GetI() * point[2] + this->GetJ();
  return (QuadricEllip <= itk::NumericTraits<ScalarType>::ZeroValue());
}

bool
QuadricShape ::IsIntersectedByRay(const PointType &  rayOrigin,
                                  const VectorType & rayDirection,
                                  ScalarType &       nearDist,
                                  ScalarType &       farDist) const
{
  // https://education.siggraph.org/static/HyperGraph/raytrace/rtinter4.htm
  ScalarType Aq = m_A * rayDirection[0] * rayDirection[0] + m_B * rayDirection[1] * rayDirection[1] +
                  m_C * rayDirection[2] * rayDirection[2] + m_D * rayDirection[0] * rayDirection[1] +
                  m_E * rayDirection[0] * rayDirection[2] + m_F * rayDirection[1] * rayDirection[2];
  ScalarType Bq = 2 * (m_A * rayOrigin[0] * rayDirection[0] + m_B * rayOrigin[1] * rayDirection[1] +
                       m_C * rayOrigin[2] * rayDirection[2]) +
                  m_D * (rayOrigin[0] * rayDirection[1] + rayOrigin[1] * rayDirection[0]) +
                  m_E * (rayOrigin[2] * rayDirection[0] + rayOrigin[0] * rayDirection[2]) +
                  m_F * (rayOrigin[1] * rayDirection[2] + rayOrigin[2] * rayDirection[1]) + m_G * rayDirection[0] +
                  m_H * rayDirection[1] + m_I * rayDirection[2];
  ScalarType Cq = m_A * rayOrigin[0] * rayOrigin[0] + m_B * rayOrigin[1] * rayOrigin[1] +
                  m_C * rayOrigin[2] * rayOrigin[2] + m_D * rayOrigin[0] * rayOrigin[1] +
                  m_E * rayOrigin[0] * rayOrigin[2] + m_F * rayOrigin[1] * rayOrigin[2] + m_G * rayOrigin[0] +
                  m_H * rayOrigin[1] + m_I * rayOrigin[2] + m_J;

  constexpr ScalarType zero = itk::NumericTraits<ScalarType>::ZeroValue();
  if (Aq == zero)
  {
    if (Bq == zero)
    {
      if (Cq == zero)
      {
        // Whole ray contained in quadric
        nearDist = -itk::NumericTraits<ScalarType>::max();
        farDist = itk::NumericTraits<ScalarType>::max();
      }
      else
        return false;
    }
    else
    {
      if (Cq == zero)
      {
        // rayOrigin is on the quadric, we need to find which half line is in
        // the quadric For t=1, we have Aq+Bq+Cq, in this case Bq. As Aq and Cq
        // are negative, we test the sign of Bq.
        if (Bq > zero)
        {
          // rayOrigin+rayDirection is not in the quadric
          farDist = zero;
          nearDist = -itk::NumericTraits<ScalarType>::max();
        }
        else
        {
          nearDist = zero;
          farDist = itk::NumericTraits<ScalarType>::max();
        }
      }
      else
      {
        // Only one intersection with, e.g., a plane: check which half line should be kept
        nearDist = -Cq / Bq;
        // The following condition tests on which side of the quadric is the ray origin.
        // It is equivalent to the following code.
        // if ((IsInsideQuadric(rayOrigin) && nearDist < 0.) ||
        //     (!IsInsideQuadric(rayOrigin) && nearDist > 0.))
        if (Cq * nearDist > zero)
          farDist = itk::NumericTraits<ScalarType>::max();
        else
        {
          farDist = nearDist;
          nearDist = -itk::NumericTraits<ScalarType>::max();
        }
      }
    }
  }
  else
  {
    ScalarType discriminant = Bq * Bq - 4 * Aq * Cq;
    // The epsilon value allows detection of very close intersections, i.e., a
    // ray tangent to the quadric
    static constexpr ScalarType eps = 1e5 * itk::NumericTraits<ScalarType>::epsilon();
    if (itk::Math::abs(discriminant) <= eps)
    {
      nearDist = -Bq / (2 * Aq);
      farDist = nearDist;
    }
    else if (discriminant < zero)
    {
      // No intersection but one might be dealing with an infinite line
      // in the quadric, e.g. a line parallel to and in a cylinder.
      // The following condition is a faster equivalent to IsInsideQuadric(rayOrigin)
      if (Cq <= zero)
      {
        nearDist = -itk::NumericTraits<ScalarType>::max();
        farDist = itk::NumericTraits<ScalarType>::max();
      }
      else
        return false;
    }
    else
    {
      nearDist = (-Bq - sqrt(discriminant)) / (2 * Aq);
      farDist = (-Bq + sqrt(discriminant)) / (2 * Aq);
      if (nearDist > farDist)
        std::swap(nearDist, farDist);

      // Check if the segment between the two intersections is in the quadric.
      // Instead of testing if the central point is oustside the quadric with
      // !IsInsideQuadric(rayOrigin + 0.5 * (farDist + nearDist) * rayDirection),
      // which was prone to numerical errors, we test if the ray origin is between
      // intersections (farDist * nearDist < 0) and strictly in the quadric (Cq < 0),
      // or before / after the two intersections (farDist * nearDist > 0) and outside
      // the quadric (Cq>0).
      if (Cq * farDist * nearDist < zero)
      {
        // If not, one is dealing with a half line, searching for the good one (out of two)
        ScalarType tmpNearDist = -itk::NumericTraits<ScalarType>::max();
        ScalarType tmpFarDist = itk::NumericTraits<ScalarType>::max();
        if (!ApplyClipPlanes(rayOrigin, rayDirection, tmpNearDist, tmpFarDist))
          return false;
        if (tmpFarDist > farDist)
        {
          if (tmpNearDist < nearDist)
          {
            itkGenericExceptionMacro(<< "Intersection of the quadric with the line "
                                     << "gives two half lines, add clip planes to select which one");
          }
          else
          {
            nearDist = std::max(farDist, tmpNearDist);
            farDist = tmpFarDist;
            return true;
          }
        }
        else if (tmpNearDist < nearDist)
        {
          farDist = std::min(nearDist, tmpFarDist);
          nearDist = tmpNearDist;
          return true;
        }
        else
          return false;
      }
    }
  }

  return ApplyClipPlanes(rayOrigin, rayDirection, nearDist, farDist);
}

void
QuadricShape ::Rescale(const VectorType & r)
{
  Superclass::Rescale(r);
  m_A /= r[0] * r[0];
  m_B /= r[1] * r[1];
  m_C /= r[2] * r[2];
  m_D /= r[0] * r[1];
  m_E /= r[0] * r[2];
  m_F /= r[1] * r[2];
  m_G /= r[0];
  m_H /= r[1];
  m_I /= r[2];
}

void
QuadricShape ::Translate(const VectorType & t)
{
  Superclass::Translate(t);

  // Translation Parameters
  ScalarType newG = m_G - 2. * m_A * t[0] - m_D * t[1] - m_E * t[2];
  ScalarType newH = m_H - 2. * m_B * t[1] - m_D * t[0] - m_F * t[2];
  ScalarType newI = m_I - 2. * m_C * t[2] - m_E * t[0] - m_F * t[1];
  ScalarType newJ = m_J + m_A * std::pow(t[0], 2.0) + m_B * std::pow(t[1], 2.0) + m_C * std::pow(t[2], 2.) +
                    m_D * t[0] * t[1] + m_E * t[0] * t[2] + m_F * t[1] * t[2] - m_G * t[0] - m_H * t[1] - m_I * t[2];
  m_G = newG;
  m_H = newH;
  m_I = newI;
  m_J = newJ;
}

void
QuadricShape ::Rotate(const RotationMatrixType & r)
{
  Superclass::Rotate(r);
  VectorType newABC, newDFE, newGHI;
  newABC.Fill(0.);
  newDFE.Fill(0.);
  newGHI.Fill(0.);
  VectorType oldABC, oldDFE, oldGHI;
  oldABC[0] = m_A;
  oldABC[1] = m_B;
  oldABC[2] = m_C;
  oldDFE[0] = m_D;
  oldDFE[1] = m_F;
  oldDFE[2] = m_E;
  oldGHI[0] = m_G;
  oldGHI[1] = m_H;
  oldGHI[2] = m_I;
  for (unsigned int j = 0; j < Dimension; j++) // Columns
  {
    for (unsigned int i = 0; i < Dimension; i++) // Lines
    {
      newABC[j] += r[j][i] * r[j][i] * oldABC[i];
      newDFE[j] += 2. * r[j][i] * r[(j + 1) % Dimension][i] * oldABC[i];
      newABC[j] += r[j][i] * r[j][(i + 1) % Dimension] * oldDFE[i];
      newDFE[j] += r[j][i] * r[(j + 1) % Dimension][(i + 1) % Dimension] * oldDFE[i];
      newDFE[j] += r[(j + 1) % Dimension][i] * r[j][(i + 1) % Dimension] * oldDFE[i];
      newGHI[j] += r[j][i] * oldGHI[i];
    }
  }
  m_A = newABC[0];
  m_B = newABC[1];
  m_C = newABC[2];
  m_D = newDFE[0];
  m_F = newDFE[1];
  m_E = newDFE[2];
  m_G = newGHI[0];
  m_H = newGHI[1];
  m_I = newGHI[2];
}

void
QuadricShape ::SetEllipsoid(const PointType & center, const VectorType & axis, const ScalarType & yangle)
{
  // A
  if (axis[0] > itk::NumericTraits<ScalarType>::ZeroValue())
    m_A = 1 / std::pow(axis[0], 2.0);
  else if (axis[0] < itk::NumericTraits<ScalarType>::ZeroValue())
    m_A = -1 / std::pow(axis[0], 2.0);
  else
    m_A = 0.;

  // B
  if (axis[1] > itk::NumericTraits<ScalarType>::ZeroValue())
    m_B = 1 / std::pow(axis[1], 2.0);
  else if (axis[1] < itk::NumericTraits<ScalarType>::ZeroValue())
    m_B = -1 / std::pow(axis[1], 2.0);
  else
    m_B = 0.;

  // C
  if (axis[2] > itk::NumericTraits<ScalarType>::ZeroValue())
    m_C = 1 / std::pow(axis[2], 2.0);
  else if (axis[2] < itk::NumericTraits<ScalarType>::ZeroValue())
    m_C = -1 / std::pow(axis[2], 2.0);
  else
    m_C = 0.;

  m_D = 0.;
  m_E = 0.;
  m_F = 0.;
  m_G = 0.;
  m_H = 0.;
  m_I = 0.;
  m_J = -1.;

  // Rotate arround Y according to angle
  ScalarType TempA = m_A;
  ScalarType TempB = m_B;
  ScalarType TempC = m_C;
  ScalarType TempG = m_G;
  ScalarType TempH = m_H;
  ScalarType TempI = m_I;
  ScalarType TempJ = m_J;

  // Applying Rotation on Y-axis
  m_A = TempA * std::pow(cos(yangle * (itk::Math::pi / 180)), 2.0) +
        TempC * std::pow(sin(yangle * (itk::Math::pi / 180)), 2.0);
  m_B = TempB;
  m_C = TempA * std::pow(sin(yangle * (itk::Math::pi / 180)), 2.0) +
        TempC * std::pow(cos(yangle * (itk::Math::pi / 180)), 2.0);
  m_D = 0.;
  m_E = 2 * cos(yangle * (itk::Math::pi / 180)) * sin(yangle * (itk::Math::pi / 180)) * (TempA - TempC);
  m_F = 0.;
  m_G = TempG * cos(yangle * (itk::Math::pi / 180)) - TempI * sin(yangle * (itk::Math::pi / 180));
  m_H = TempH;
  m_I = TempG * sin(yangle * (itk::Math::pi / 180)) + TempI * cos(yangle * (itk::Math::pi / 180));
  m_J = TempJ;

  Translate(center.GetVectorFromOrigin());
}

itk::LightObject::Pointer
QuadricShape ::InternalClone() const
{
  LightObject::Pointer loPtr = Superclass::InternalClone();
  Self::Pointer        clone = dynamic_cast<Self *>(loPtr.GetPointer());

  clone->SetA(this->GetA());
  clone->SetB(this->GetB());
  clone->SetC(this->GetC());
  clone->SetD(this->GetD());
  clone->SetE(this->GetE());
  clone->SetF(this->GetF());
  clone->SetG(this->GetG());
  clone->SetH(this->GetH());
  clone->SetI(this->GetI());
  clone->SetJ(this->GetJ());

  return loPtr;
}
} // end namespace rtk
