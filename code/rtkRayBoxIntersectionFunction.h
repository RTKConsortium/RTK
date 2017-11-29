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

#ifndef rtkRayBoxIntersectionFunction_h
#define rtkRayBoxIntersectionFunction_h

#include <itkNumericTraits.h>
#include <vector>
#include <itkImageBase.h>
#include "rtkConfiguration.h"

namespace rtk
{

/** \class RayBoxIntersectionFunction
 * \brief Compute the intersection between a ray and a box.
 *
 * The box is defined by two corners and the orientation can be set similarly
 * to an itk::Image, by setting the m_Direction member. The ray origin must be
 * set first. The direction of the ray is then passed to the Evaluate function.
 * It returns false if there is no intersection. It returns true otherwise and
 * the nearest and farthest distance/point may be accessed. Nearest and
 * farthest distances are defined such that NearestDistance < FarthestDistance.
 *
 * The default behavior of the function is to return the intersection between
 * the line defined by the origin and direction. You need to modify the
 * nearest and farthest distance if you want to account for the position of the
 * source and the detector along the ray.
 *
 * \author Simon Rit
 *
 * \ingroup Functions
 */
template <
class TCoordRep = double,
unsigned int VBoxDimension=3
>
class ITK_EXPORT RayBoxIntersectionFunction:
    public itk::LightObject
{
public:
  /** Standard class typedefs. */
  typedef RayBoxIntersectionFunction    Self;
  typedef itk::Object                   Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Useful defines. */
  typedef itk::Vector<TCoordRep, VBoxDimension>                                VectorType;
  typedef itk::ImageBase<VBoxDimension>                                        ImageBaseType;
  typedef itk::Matrix< itk::SpacePrecisionType, VBoxDimension, VBoxDimension > DirectionType;

  /** Evaluate the intersection points */
  bool Evaluate( const VectorType& input );

  /** Set the box information (Min/Max corners) from an itk image.
   * \warning this method caches image information.
   * If the image information has changed, user must call
   * SetBoxFromImage again to update cached values.
   * bWithExternalHalfPixelBorder adds a half pixel border around the image, the
   * box stops at the center of the first and the last voxel otherwise. */
  void SetBoxFromImage( const ImageBaseType *img, bool bWithExternalHalfPixelBorder=true );

  /** Get / Set the box inferior corner. Every coordinate must be inferior to
   * those of the superior corner. */
  virtual VectorType GetBoxMin();
  virtual void SetBoxMin(const VectorType _arg);

  /** Get / Set the box superior corner. Every coordinate must be superior to
   * those of the inferior corner. */
  virtual VectorType GetBoxMax();
  virtual void SetBoxMax(const VectorType _arg);

  /** Direction is the direction of the box, defined in the same sense as in
    * itk::ImageBase. */
  virtual DirectionType GetDirection();
  virtual void SetDirection(const DirectionType _arg);

  /** Get / Set the ray origin. */
  virtual VectorType GetRayOrigin() { return this->m_RayOrigin; }
  virtual void SetRayOrigin(const VectorType _arg) { m_RayOrigin = _arg; }

  /** Get the distance with the nearest intersection.
    * \warning Only relevant if called after Evaluate. */
  virtual TCoordRep GetNearestDistance() { return this->m_NearestDistance; }
  virtual void SetNearestDistance(const TCoordRep _arg) { m_NearestDistance = _arg; }

  /** Get the distance with the farthest intersection.
    * \warning Only relevant if called after Evaluate. */
  virtual TCoordRep GetFarthestDistance() { return this->m_FarthestDistance; }
  virtual void SetFarthestDistance(const TCoordRep _arg) { m_FarthestDistance = _arg; }

  /** Get the nearest point coordinates.
    * \warning Only relevant if called after Evaluate. */
  virtual VectorType GetNearestPoint()
  {
    return m_RayOrigin + m_NearestDistance * m_RayDirection;
  }

  /** Get the farthest point coordinates.
    * \warning Only relevant if called after Evaluate. */
  virtual VectorType GetFarthestPoint()
  {
    return m_RayOrigin + m_FarthestDistance * m_RayDirection;
  }

protected:

  RayBoxIntersectionFunction();

  ~RayBoxIntersectionFunction() {}

  void TransformBoxCornersWithDirection();

  /// The focal point or position of the ray source
  VectorType m_FocalPoint;

  /** Corners of the image box */
  VectorType    m_BoxMin;
  VectorType    m_BoxMax;
  VectorType    m_BoxMinT;
  VectorType    m_BoxMaxT;
  VectorType    m_RayOrigin;
  VectorType    m_RayDirection;
  TCoordRep     m_NearestDistance;
  TCoordRep     m_FarthestDistance;
  DirectionType m_Direction;
  DirectionType m_DirectionT;

private:
  RayBoxIntersectionFunction( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented
};

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayBoxIntersectionFunction.hxx"
#endif

#endif
