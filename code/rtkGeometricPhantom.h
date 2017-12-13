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

#include "rtkConvexObject.h"

namespace rtk
{
/** \class GeometricPhantom
 * \brief Geometric phantom, i.e., a set of rtk::ConvexObjects.
 *
 * \author Simon Rit
 *
 */
class RTK_EXPORT GeometricPhantom: public itk::DataObject
{
public:
  /** Standard class typedefs. */
  typedef GeometricPhantom              Self;
  typedef itk::DataObject               Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Convenient typedefs. */
  typedef ConvexObject::Pointer            ConvexObjectPointer;
  typedef std::vector<ConvexObjectPointer> ConvexObjectVector;
  typedef ConvexObject::PointType          PointType;
  typedef ConvexObject::VectorType         VectorType;
  typedef ConvexObject::RotationMatrixType RotationMatrixType;


  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GeometricPhantom, itk::DataObject);

  /** Rescale object along each direction by a 3D vector. */
  virtual void Rescale(const VectorType &r);

  /** Translate object by a given 3D vector. */
  virtual void Translate(const VectorType &t);

  /** Translate object by a given 3D vector. */
  virtual void Rotate(const RotationMatrixType &r);

  /** Get reference to vector of objects. */
  itkGetConstReferenceMacro(ConvexObjects, ConvexObjectVector);

  /** Add convex object to phantom. */
  void AddConvexObject(const ConvexObjectPointer &co);

protected:
  GeometricPhantom() {}
  ~GeometricPhantom() {}

private:
  GeometricPhantom(const Self&); //purposely not implemented
  void operator=(const Self&);   //purposely not implemented

  ConvexObjectVector m_ConvexObjects;
};

}
#endif // rtkGeometricPhantom_h
