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

#ifndef rtkConvexObject_h
#define rtkConvexObject_h

#include <itkMatrix.h>
#include <itkPoint.h>
#include <itkDataObject.h>
#include <itkObjectFactory.h>

#include "rtkMacro.h"
#include "rtkWin32Header.h"

namespace rtk
{
/** \class ConvexObject
 * \brief Base class for a 3D object. rtk::DrawImageFilter fills uses it (and IsInside) to fill a volume.
 *
 * \author Mathieu Dupont, Simon Rit
 *
 */
class RTK_EXPORT ConvexObject: public itk::DataObject
{
public:
  /** Standard class typedefs. */
  typedef ConvexObject                  Self;
  typedef itk::DataObject               Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Convenient typedefs. */
  itkStaticConstMacro(Dimension, unsigned int, 3);
  typedef double                                          ScalarType;
  typedef itk::Vector< ScalarType, Dimension >            PointType;
  typedef itk::Vector< ScalarType, Dimension >            VectorType;
  typedef itk::Matrix< ScalarType, Dimension, Dimension > RotationMatrixType;

  /** Method for creation through the object factory. */
  itkNewMacro ( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConvexObject, itk::DataObject);

  /** Returns true if a point is inside the object. */
  virtual bool IsInside(const PointType & point) const;

  /** Returns true if a ray intersects the object. If it does, you can access
   * the intersection points with GetNearestDistance() and GetFarthestDistance. */
  virtual bool IsIntersectedByRay(const PointType & rayOrigin,
                                  const VectorType & rayDirection,
                                  double & near,
                                  double & far) const;

  /** Rescale object along each direction by a 3D vector. */
  virtual void Rescale(const VectorType &r);

  /** Translate object by a given 3D vector. */
  virtual void Translate(const VectorType &t);

  /** Translate object by a given 3D vector. */
  virtual void Rotate(const RotationMatrixType &r);

  /** Add clipping plane to the object. The plane is defined by the equation
   * dir * (x,y,z)' + pos = 0. */
  void AddClippingPlane(const VectorType & dir, const ScalarType & pos);
  void SetClippingPlanes(const std::vector<VectorType> & dir, const std::vector<ScalarType> & pos);

  /** Volume density, i.e., value in the volume. */
  itkSetMacro (Density, ScalarType);
  itkGetConstMacro (Density, ScalarType);
  itkGetMacro (Density, ScalarType);

  /** Get reference to vector of plane parameters. */
  itkGetConstReferenceMacro(PlaneDirections, std::vector<VectorType>);
  itkGetConstReferenceMacro(PlanePositions, std::vector<ScalarType>);

protected:
  ConvexObject();
  bool ApplyClippingPlanes(const PointType & rayOrigin,
                           const VectorType & rayDirection,
                           ScalarType & near,
                           ScalarType & far) const;
  bool ApplyClippingPlanes(const PointType & point) const;
  virtual itk::LightObject::Pointer InternalClone() const ITK_OVERRIDE;

private:
  ConvexObject(const Self&);   //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  ScalarType              m_Density;
  std::vector<VectorType> m_PlaneDirections;
  std::vector<ScalarType> m_PlanePositions;
};

}
#endif
