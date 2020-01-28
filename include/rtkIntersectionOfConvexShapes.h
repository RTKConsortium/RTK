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

#ifndef rtkIntersectionOfConvexShapes_h
#define rtkIntersectionOfConvexShapes_h

#include "RTKExport.h"
#include "rtkMacro.h"
#include "rtkConvexShape.h"

namespace rtk
{

/** \class IntersectionOfConvexShapes
 * \brief Defines a shape as the intersection of several ConvexShape
 *
 * \test rtkforbildtest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
class RTK_EXPORT IntersectionOfConvexShapes : public ConvexShape
{
public:
  /** Standard class type alias. */
  using Self = IntersectionOfConvexShapes;
  using Superclass = ConvexShape;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  static constexpr unsigned int Dimension = Superclass::Dimension;
  using ConvexShapePointer = ConvexShape::Pointer;
  using ConvexShapeVector = std::vector<ConvexShapePointer>;
  using ScalarType = Superclass::ScalarType;
  using PointType = Superclass::PointType;
  using VectorType = Superclass::VectorType;
  using RotationMatrixType = Superclass::RotationMatrixType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(IntersectionOfConvexShapes, ConvexShape);

  /** See rtk::ConvexShape::IsInside. */
  bool
  IsInside(const PointType & point) const override;

  /** See rtk::ConvexShape::IsIntersectedByRay. */
  bool
  IsIntersectedByRay(const PointType &  rayOrigin,
                     const VectorType & rayDirection,
                     ScalarType &       nearDist,
                     ScalarType &       farDist) const override;

  /** Add convex object to phantom. */
  void
  AddConvexShape(const ConvexShape *co);
  itkGetConstReferenceMacro(ConvexShapes, ConvexShapeVector);
  virtual void
  SetConvexShapes(const ConvexShapeVector & _arg);

  /** Rescale object along each direction by a 3D vector. */
  void
  Rescale(const VectorType & r) override;

  /** Translate object by a given 3D vector. */
  void
  Translate(const VectorType & t) override;

  /** Rotate object according to a 3D rotation matrix. */
  void
  Rotate(const RotationMatrixType & r) override;

  itk::LightObject::Pointer
  InternalClone() const override;

private:
  IntersectionOfConvexShapes();

  ConvexShapeVector m_ConvexShapes;
};

} // end namespace rtk

#endif
