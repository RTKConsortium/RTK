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

#ifndef rtkBoxShape_h
#define rtkBoxShape_h

#include "RTKExport.h"
#include "rtkMacro.h"
#include "rtkConvexShape.h"

#include <itkImageBase.h>

namespace rtk
{

/** \class BoxShape
 * \brief BoxShape defines a paralleliped.
 * The box is defined by its two opposite corners, BoxMin and BoxMax, and a
 * rotation matrix Direction. The box corresponding to an Image can be set
 * using the function SetBoxShapeFromImage.
 *
 * \test rtkforbildtest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 *
 */
class RTK_EXPORT BoxShape : public ConvexShape
{
public:
  /** Standard class type alias. */
  using Self = BoxShape;
  using Superclass = ConvexShape;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  static constexpr unsigned int Dimension = Superclass::Dimension;
  using ScalarType = Superclass::ScalarType;
  using PointType = Superclass::PointType;
  using VectorType = Superclass::VectorType;
  using RotationMatrixType = Superclass::RotationMatrixType;
  using ImageBaseType = itk::ImageBase<Dimension>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BoxShape, ConvexShape);

  /** See rtk::ConvexShape::IsInside. */
  bool
  IsInside(const PointType & point) const override;

  /** See rtk::ConvexShape::IsIntersectedByRay for the goal and
   * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
   * for the computation. */
  bool
  IsIntersectedByRay(const PointType &  rayOrigin,
                     const VectorType & rayDirection,
                     double &           nearDist,
                     double &           farDist) const override;

  /** Rescale object along each direction by a 3D vector. */
  void
  Rescale(const VectorType & r) override;

  /** Translate object by a given 3D vector. */
  void
  Translate(const VectorType & t) override;

  /** Translate object by a given 3D vector. */
  void
  Rotate(const RotationMatrixType & r) override;

  /** Get / Set the box inferior corner. Every corner coordinate must be
   * inferior to those of the superior corner. */
  itkGetConstMacro(BoxMin, PointType);
  itkSetMacro(BoxMin, PointType);

  /** Get / Set the box superior corner. Every corner coordinate must be
   * superior to those of the inferior corner. */
  itkGetConstMacro(BoxMax, PointType);
  itkSetMacro(BoxMax, PointType);

  /** Direction is the direction of the box, defined in the same sense as in
   * itk::ImageBase. */
  itkGetConstMacro(Direction, RotationMatrixType);
  itkSetMacro(Direction, RotationMatrixType);

  itk::LightObject::Pointer
  InternalClone() const override;

  /** Set the 3D box is the portion of space defined by the LargestPossibleRegion.
   * bWithExternalHalfPixelBorder can be used to include or exclude a half voxel
   * border. */
  void
  SetBoxFromImage(const ImageBaseType * img, bool bWithExternalHalfPixelBorder = true);

private:
  BoxShape();

  PointType          m_BoxMin;
  PointType          m_BoxMax;
  RotationMatrixType m_Direction;
};

} // end namespace rtk

#endif
