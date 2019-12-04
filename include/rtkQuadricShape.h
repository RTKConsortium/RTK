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

#ifndef rtkQuadricShape_h
#define rtkQuadricShape_h

#include "RTKExport.h"
#include <itkPoint.h>
#include <rtkMacro.h>

#include "rtkConvexShape.h"

namespace rtk
{

/** \class QuadricShape
 * \brief Defines a 3D quadric shape.
 *
 * A quadric shape has the equation
 * Ax^2 + By^2 + Cz^2 + Dxy+ Exz + Fyz + Gx + Hy + Iz + J = 0
 * It is assumed to be convex (which is not always true).
 *
 * \test rtkforbildtest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 *
 */
class RTK_EXPORT QuadricShape : public ConvexShape
{
public:
  /** Standard class type alias. */
  using Self = QuadricShape;
  using Superclass = ConvexShape;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  using ScalarType = Superclass::ScalarType;
  using PointType = Superclass::PointType;
  using VectorType = Superclass::VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(QuadricShape, ConvexShape);

  /** See rtk::ConvexShape::IsInside. */
  bool
  IsInside(const PointType & point) const override;

  /** See rtk::ConvexShape::IsIntersectedByRay for the goal and
   * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
   * for the computation. */
  bool
  IsIntersectedByRay(const PointType &  rayOrigin,
                     const VectorType & rayDirection,
                     double &           nearDist,
                     double &           farDist) const override;

  /** Rescale object along each direction by a 3D vector. */
  void
  Rescale(const VectorType & r) override;

  /** Translate object by a given 3D vector. */
  void
  Translate(const VectorType & t) override;

  /** Rotate object by a given 3D vector. */
  void
  Rotate(const RotationMatrixType & r) override;

  itkGetConstMacro(A, ScalarType);
  itkSetMacro(A, ScalarType);
  itkGetConstMacro(B, ScalarType);
  itkSetMacro(B, ScalarType);
  itkGetConstMacro(C, ScalarType);
  itkSetMacro(C, ScalarType);
  itkGetConstMacro(D, ScalarType);
  itkSetMacro(D, ScalarType);
  itkGetConstMacro(E, ScalarType);
  itkSetMacro(E, ScalarType);
  itkGetConstMacro(F, ScalarType);
  itkSetMacro(F, ScalarType);
  itkGetConstMacro(G, ScalarType);
  itkSetMacro(G, ScalarType);
  itkGetConstMacro(H, ScalarType);
  itkSetMacro(H, ScalarType);
  itkGetConstMacro(I, ScalarType);
  itkSetMacro(I, ScalarType);
  itkGetConstMacro(J, ScalarType);
  itkSetMacro(J, ScalarType);

  void
  SetEllipsoid(const PointType & center, const VectorType & axis, const ScalarType & yangle = 0);

  itk::LightObject::Pointer
  InternalClone() const override;

private:
  QuadricShape();

  ScalarType m_A{ 0. };
  ScalarType m_B{ 0. };
  ScalarType m_C{ 0. };
  ScalarType m_D{ 0. };
  ScalarType m_E{ 0. };
  ScalarType m_F{ 0. };
  ScalarType m_G{ 0. };
  ScalarType m_H{ 0. };
  ScalarType m_I{ 0. };
  ScalarType m_J{ 0. };
};

} // end namespace rtk

#endif
