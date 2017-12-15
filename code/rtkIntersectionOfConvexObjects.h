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

#ifndef rtkIntersectionOfConvexObjects_h
#define rtkIntersectionOfConvexObjects_h

#include "rtkWin32Header.h"
#include "rtkMacro.h"
#include "rtkConvexObject.h"

namespace rtk
{

class RTK_EXPORT IntersectionOfConvexObjects:
    public ConvexObject
{
public:
  /** Standard class typedefs. */
  typedef IntersectionOfConvexObjects   Self;
  typedef ConvexObject                  Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Convenient typedefs. */
  itkStaticConstMacro(Dimension, unsigned int, Superclass::Dimension);
  typedef ConvexObject::Pointer            ConvexObjectPointer;
  typedef std::vector<ConvexObjectPointer> ConvexObjectVector;
  typedef Superclass::ScalarType           ScalarType;
  typedef Superclass::PointType            PointType;
  typedef Superclass::VectorType           VectorType;
  typedef Superclass::RotationMatrixType   RotationMatrixType;

  /** Method for creation through the object factory. */
  itkNewMacro ( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro(IntersectionOfConvexObjects, ConvexObject);

  /** See rtk::ConvexObject::IsInside. */
  virtual bool IsInside(const PointType & point) const ITK_OVERRIDE;

  /** See rtk::ConvexObject::IsIntersectedByRay for the goal and
   * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
   * for the computation. */
  virtual bool IsIntersectedByRay(const PointType & rayOrigin,
                                  const VectorType & rayDirection,
                                  ScalarType & near,
                                  ScalarType & far) const ITK_OVERRIDE;

  /** Add convex object to phantom. */
  void AddConvexObject(const ConvexObjectPointer &co);
  itkGetConstReferenceMacro(ConvexObjects, ConvexObjectVector);
  itkSetMacro(ConvexObjects, ConvexObjectVector);

  /** Rescale object along each direction by a 3D vector. */
  virtual void Rescale(const VectorType &r) ITK_OVERRIDE;

  /** Translate object by a given 3D vector. */
  virtual void Translate(const VectorType &t) ITK_OVERRIDE;

  /** Translate object by a given 3D vector. */
  virtual void Rotate(const RotationMatrixType &r) ITK_OVERRIDE;

  virtual itk::LightObject::Pointer InternalClone() const ITK_OVERRIDE;

private:
  IntersectionOfConvexObjects();

  ConvexObjectVector m_ConvexObjects;
};

} // end namespace rtk

#endif
