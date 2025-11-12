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

#ifndef rtkConvexShape_h
#define rtkConvexShape_h

#include <itkMatrix.h>
#include <itkPoint.h>
#include <itkDataObject.h>
#include <itkObjectFactory.h>

#include "RTKExport.h"
#include "rtkMacro.h"

namespace rtk
{
/** \class ConvexShape
 * \brief Base class for a 3D convex shape.
 *
 * A ConvexShape is used to draw and project (in the tomographic sense) a
 * geometric phantom using the functions IsInside and IsIntersectedByRay,
 * respectively.
 *
 * \test rtkforbildtest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 *
 */
class RTK_EXPORT ConvexShape : public itk::DataObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ConvexShape);

  /** Standard class type alias. */
  using Self = ConvexShape;
  using Superclass = itk::DataObject;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  static constexpr unsigned int Dimension = 3;
  using ScalarType = double;
  using PointType = itk::Point<ScalarType, Dimension>;
  using VectorType = itk::Vector<ScalarType, Dimension>;
  using RotationMatrixType = itk::Matrix<ScalarType, Dimension, Dimension>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ConvexShape);

  /** Returns true if a point is inside the object. */
  virtual bool
  IsInside(const PointType & point) const;

  /** Returns true if a ray intersects the object. If it does, the parameters
  ** nearDist and farDist get the shape distance from the source in the ray direction.
  ** Note that nearDist<farDist, and nearDist and farDist can be negative. */
  virtual bool
  IsIntersectedByRay(const PointType &  rayOrigin,
                     const VectorType & rayDirection,
                     ScalarType &       nearDist,
                     ScalarType &       farDist) const;

  /** Rescale object along each direction by a 3D vector. */
  virtual void
  Rescale(const VectorType & r);

  /** Translate object by a given 3D vector. */
  virtual void
  Translate(const VectorType & t);

  /** Rotate object according to a 3D rotation matrix. */
  virtual void
  Rotate(const RotationMatrixType & r);

  /** Add clipping plane to the object. The plane is defined by the equation
   * dir * (x,y,z)' + pos = 0. */
  void
  AddClipPlane(const VectorType & dir, const ScalarType & pos);
  void
  SetClipPlanes(const std::vector<VectorType> & dir, const std::vector<ScalarType> & pos);

  /** Volume density, i.e., value in the volume. */
  itkSetMacro(Density, ScalarType);
  itkGetConstMacro(Density, ScalarType);
  itkGetMacro(Density, ScalarType);

  /** Get reference to vector of plane parameters. */
  itkGetConstReferenceMacro(PlaneDirections, std::vector<VectorType>);
  itkGetConstReferenceMacro(PlanePositions, std::vector<ScalarType>);

protected:
  ConvexShape();
  bool
  ApplyClipPlanes(const PointType &  rayOrigin,
                  const VectorType & rayDirection,
                  ScalarType &       nearDist,
                  ScalarType &       farDist) const;
  bool
  ApplyClipPlanes(const PointType & point) const;
  itk::LightObject::Pointer
  InternalClone() const override;

private:
  ScalarType              m_Density{ 0. };
  std::vector<VectorType> m_PlaneDirections;
  std::vector<ScalarType> m_PlanePositions;
};

} // namespace rtk
#endif
