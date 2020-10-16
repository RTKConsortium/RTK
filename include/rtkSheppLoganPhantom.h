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
 * \brief SheppLogan phantom as defined in "Principles of CT imaging" by Kak & Slaney
 *
 * See http://www.slaney.org/pct/pct-errata.html for the correction of the
 * phantom definition.
 *
 * \test rtkfdktest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 *
 */
class RTK_EXPORT SheppLoganPhantom : public GeometricPhantom
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(SheppLoganPhantom);
#else
  ITK_DISALLOW_COPY_AND_MOVE(SheppLoganPhantom);
#endif

  /** Standard class type alias. */
  using Self = SheppLoganPhantom;
  using Superclass = itk::DataObject;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  using ScalarType = QuadricShape::ScalarType;
  using PointType = QuadricShape::PointType;
  using VectorType = QuadricShape::VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SheppLoganPhantom, GeometricPhantom);

protected:
  SheppLoganPhantom();
  ~SheppLoganPhantom() override = default;

private:
  void
  SetEllipsoid(ScalarType spax,
               ScalarType spay,
               ScalarType spaz,
               ScalarType centerx,
               ScalarType centery,
               ScalarType centerz,
               ScalarType angle,
               ScalarType density);
};

} // namespace rtk
#endif // rtkSheppLoganPhantom_h
