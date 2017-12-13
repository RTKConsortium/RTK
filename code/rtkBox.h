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

#ifndef rtkBox_h
#define rtkBox_h

#include "rtkWin32Header.h"
#include "rtkMacro.h"
#include "rtkConvexObject.h"

#include <itkImageBase.h>

namespace rtk
{

class RTK_EXPORT Box:
    public ConvexObject
{
public:
  /** Standard class typedefs. */
  typedef Box                           Self;
  typedef ConvexObject                  Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Convenient typedefs. */
  itkStaticConstMacro(Dimension, unsigned int, Superclass::Dimension);
  typedef Superclass::ScalarType         ScalarType;
  typedef Superclass::PointType          PointType;
  typedef Superclass::VectorType         VectorType;
  typedef Superclass::RotationMatrixType RotationMatrixType;
  typedef itk::ImageBase<Dimension>      ImageBaseType;

  /** Method for creation through the object factory. */
  itkNewMacro ( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro(Box, ConvexObject);

  /** See rtk::ConvexObject::IsInside. */
  virtual bool IsInside(const PointType & point) const ITK_OVERRIDE;

  /** See rtk::ConvexObject::IsIntersectedByRay for the goal and
   * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
   * for the computation. */
  virtual bool IsIntersectedByRay(const PointType & rayOrigin,
                                  const VectorType & rayDirection,
                                  double &near,
                                  double &far) const ITK_OVERRIDE;

  /** Rescale object along each direction by a 3D vector. */
  virtual void Rescale(const VectorType &r) ITK_OVERRIDE;

  /** Translate object by a given 3D vector. */
  virtual void Translate(const VectorType &t) ITK_OVERRIDE;

  /** Translate object by a given 3D vector. */
  virtual void Rotate(const RotationMatrixType &r) ITK_OVERRIDE;

    /** Get / Set the box inferior corner. Every coordinate must be inferior to
   * those of the superior corner. */
  itkGetConstMacro(BoxMin, PointType);
  itkSetMacro(BoxMin, PointType);

  /** Get / Set the box superior corner. Every coordinate must be superior to
   * those of the inferior corner. */
  itkGetConstMacro(BoxMax, PointType);
  itkSetMacro(BoxMax, PointType);

  /** Direction is the direction of the box, defined in the same sense as in
    * itk::ImageBase. */
  itkGetConstMacro(Direction, RotationMatrixType);
  itkSetMacro(Direction, RotationMatrixType);

  virtual itk::LightObject::Pointer InternalClone() const ITK_OVERRIDE;

  void SetBoxFromImage( const ImageBaseType *img, bool bWithExternalHalfPixelBorder=true );

private:
  Box();

  PointType          m_BoxMin;
  PointType          m_BoxMax;
  RotationMatrixType m_Direction;
};

} // end namespace rtk

#endif
