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

#ifndef rtkSheppLoganPhantom_h
#define rtkSheppLoganPhantom_h

#include "rtkGeometricPhantom.h"
#include "rtkQuadricShape.h"

namespace rtk
{
/** \class SheppLoganPhantom
 * \brief SheppLogan phantom, i.e., a set of rtk::ConvexObjects.
 *
 * \author Simon Rit
 *
 */
class RTK_EXPORT SheppLoganPhantom: public GeometricPhantom
{
public:
  /** Standard class typedefs. */
  typedef SheppLoganPhantom             Self;
  typedef itk::DataObject               Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Convenient typedefs. */
  typedef QuadricShape::ScalarType ScalarType;
  typedef QuadricShape::PointType  PointType;
  typedef QuadricShape::VectorType VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SheppLoganPhantom, GeometricPhantom);

protected:
  SheppLoganPhantom();
  ~SheppLoganPhantom() {}

private:
  SheppLoganPhantom(const Self&); //purposely not implemented
  void operator=(const Self&);   //purposely not implemented

  void SetEllipsoid(ScalarType spax,    ScalarType spay,    ScalarType spaz,
                    ScalarType centerx, ScalarType centery, ScalarType centerz,
                    ScalarType angle, ScalarType density);
};

}
#endif // rtkSheppLoganPhantom_h
