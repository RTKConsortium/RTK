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

#include "rtkWin32Header.h"
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
 */
class RTK_EXPORT IntersectionOfConvexShapes:
    public ConvexShape
{
public:
  /** Standard class typedefs. */
  typedef IntersectionOfConvexShapes    Self;
  typedef ConvexShape                   Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Convenient typedefs. */
  itkStaticConstMacro(Dimension, unsigned int, Superclass::Dimension);
  typedef ConvexShape::Pointer             ConvexShapePointer;
  typedef std::vector<ConvexShapePointer>  ConvexShapeVector;
  typedef Superclass::ScalarType           ScalarType;
  typedef Superclass::PointType            PointType;
  typedef Superclass::VectorType           VectorType;
  typedef Superclass::RotationMatrixType   RotationMatrixType;

  /** Method for creation through the object factory. */
  itkNewMacro ( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro(IntersectionOfConvexShapes, ConvexShape);

  /** See rtk::ConvexShape::IsInside. */
  virtual bool IsInside(const PointType & point) const ITK_OVERRIDE;

  /** See rtk::ConvexShape::IsIntersectedByRay. */
  virtual bool IsIntersectedByRay(const PointType & rayOrigin,
                                  const VectorType & rayDirection,
                                  ScalarType & nearDist,
                                  ScalarType & farDist) const ITK_OVERRIDE;

  /** Add convex object to phantom. */
  void AddConvexShape(const ConvexShapePointer &co);
  itkGetConstReferenceMacro(ConvexShapes, ConvexShapeVector);
  itkSetMacro(ConvexShapes, ConvexShapeVector);

  /** Rescale object along each direction by a 3D vector. */
  virtual void Rescale(const VectorType &r) ITK_OVERRIDE;

  /** Translate object by a given 3D vector. */
  virtual void Translate(const VectorType &t) ITK_OVERRIDE;

  /** Rotate object according to a 3D rotation matrix. */
  virtual void Rotate(const RotationMatrixType &r) ITK_OVERRIDE;

  virtual itk::LightObject::Pointer InternalClone() const ITK_OVERRIDE;

private:
  IntersectionOfConvexShapes();

  ConvexShapeVector m_ConvexShapes;
};

} // end namespace rtk

#endif
