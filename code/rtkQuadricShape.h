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

#include "rtkWin32Header.h"

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
 */
class RTK_EXPORT QuadricShape:
    public ConvexShape
{
public:
  /** Standard class typedefs. */
  typedef QuadricShape                  Self;
  typedef ConvexShape                   Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Convenient typedefs. */
  typedef Superclass::ScalarType ScalarType;
  typedef Superclass::PointType  PointType;
  typedef Superclass::VectorType VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro ( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro(QuadricShape, ConvexShape);

  /** See rtk::ConvexShape::IsInside. */
  virtual bool IsInside(const PointType & point) const ITK_OVERRIDE;

  /** See rtk::ConvexShape::IsIntersectedByRay for the goal and
   * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
   * for the computation. */
  virtual bool IsIntersectedByRay(const PointType & rayOrigin,
                                  const VectorType & rayDirection,
                                  double & near,
                                  double & far) const ITK_OVERRIDE;

  /** Rescale object along each direction by a 3D vector. */
  virtual void Rescale(const VectorType &r) ITK_OVERRIDE;

  /** Translate object by a given 3D vector. */
  virtual void Translate(const VectorType &t) ITK_OVERRIDE;

  /** Rotate object by a given 3D vector. */
  virtual void Rotate(const RotationMatrixType &r) ITK_OVERRIDE;

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

  void SetEllipsoid(const PointType &center, const VectorType &axis, const ScalarType &yangle=0);

  virtual itk::LightObject::Pointer InternalClone() const ITK_OVERRIDE;

private:
  QuadricShape();

  ScalarType m_A;
  ScalarType m_B;
  ScalarType m_C;
  ScalarType m_D;
  ScalarType m_E;
  ScalarType m_F;
  ScalarType m_G;
  ScalarType m_H;
  ScalarType m_I;
  ScalarType m_J;
};

} // end namespace rtk

#endif
