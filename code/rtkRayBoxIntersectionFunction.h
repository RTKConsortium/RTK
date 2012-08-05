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

#ifndef __rtkRayBoxIntersectionFunction_h
#define __rtkRayBoxIntersectionFunction_h

#include <itkNumericTraits.h>
#include <vector>
#include <itkImageBase.h>

namespace rtk
{

/** \class RayBoxIntersectionFunction
 * \brief Test if a ray intersects with a box.
 *
 * TODO
 *
 * \author Simon Rit
 *
 * \ingroup Functions
 */
template <
class TCoordRep = double,
unsigned int VBoxDimension=3
>
class ITK_EXPORT RayBoxIntersectionFunction :
    public itk::Object
{
public:
  /** Standard class typedefs. */
  typedef RayBoxIntersectionFunction    Self;
  typedef itk::Object                   Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  typedef typename itk::ImageBase<VBoxDimension>::ConstPointer ImageBaseConstPointer;


  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Useful defines. */
  typedef itk::Vector<TCoordRep, VBoxDimension> VectorType;

  /** Evaluate the intersection points */
  bool Evaluate( const VectorType& input );

  /** Set the box information (Min/Max corners) from an itk image.
   * \warning this method caches image information.
   * If the image information has changed, user must call
   * SetBoxFromImage again to update cached values. */
  void SetBoxFromImage( ImageBaseConstPointer img );

  /** Get / Set the box inferior corner. Every coordinate must be inferior to
   * those of the superior corner. */
  itkGetMacro(BoxMin, VectorType);
  itkSetMacro(BoxMin, VectorType);

  /** Get / Set the box superior corner. Every coordinate must be superior to
   * those of the inferior corner. */
  itkGetMacro(BoxMax, VectorType);
  itkSetMacro(BoxMax, VectorType);

  /** Get / Set the ray origin. */
  itkGetMacro(RayOrigin, VectorType);
  itkSetMacro(RayOrigin, VectorType);

  /** Get the distance with the nearest intersection.
    * \warning Only relevant if called after Evaluate. */
  itkGetMacro(NearestDistance, TCoordRep);

  /** Get the distance with the farthest intersection.
    * \warning Only relevant if called after Evaluate. */
  itkGetMacro(FarthestDistance, TCoordRep);

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

  /// Constructor
  RayBoxIntersectionFunction(){};

  /// Destructor
  ~RayBoxIntersectionFunction(){};

  /// The focal point or position of the ray source
  VectorType m_FocalPoint;

  /** Corners of the image box */
  VectorType m_BoxMin;
  VectorType m_BoxMax;
  VectorType m_RayOrigin;
  VectorType m_RayDirection;
  TCoordRep  m_NearestDistance;
  TCoordRep  m_FarthestDistance;

private:
  RayBoxIntersectionFunction( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented
};

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayBoxIntersectionFunction.txx"
#endif

#endif
