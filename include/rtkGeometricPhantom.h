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

#ifndef rtkGeometricPhantom_h
#define rtkGeometricPhantom_h

#include "rtkConvexShape.h"

namespace rtk
{
/** \class GeometricPhantom
 * \brief Container for a geometric phantom, i.e., a set of ConvexShapes.
 *
 * \test rtkforbildtest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 *
 */
class RTK_EXPORT GeometricPhantom : public itk::DataObject
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(GeometricPhantom);

  /** Standard class type alias. */
  using Self = GeometricPhantom;
  using Superclass = itk::DataObject;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  using ConvexShapePointer = ConvexShape::Pointer;
  using ConvexShapeVector = std::vector<ConvexShapePointer>;
  using PointType = ConvexShape::PointType;
  using VectorType = ConvexShape::VectorType;
  using ScalarType = ConvexShape::ScalarType;
  using RotationMatrixType = ConvexShape::RotationMatrixType;


  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GeometricPhantom, itk::DataObject);

  /** Rescale object along each direction by a 3D vector. */
  virtual void
  Rescale(const VectorType & r);

  /** Translate object by a given 3D vector. */
  virtual void
  Translate(const VectorType & t);

  /** Rotate object according to a 3D rotation matrix. */
  virtual void
  Rotate(const RotationMatrixType & r);

  /** Get reference to vector of objects. */
  itkGetConstReferenceMacro(ConvexShapes, ConvexShapeVector);

  /** Add convex object to phantom. */
  void
  AddConvexShape(const ConvexShape *co);

  /** Add clipping plane to the object. The plane is defined by the equation
   * dir * (x,y,z)' + pos = 0. */
  void
  AddClipPlane(const VectorType & dir, const ScalarType & pos);

protected:
  GeometricPhantom() = default;
  ~GeometricPhantom() override = default;

private:
  ConvexShapeVector       m_ConvexShapes;
  std::vector<VectorType> m_PlaneDirections;
  std::vector<ScalarType> m_PlanePositions;
};

} // namespace rtk
#endif // rtkGeometricPhantom_h
