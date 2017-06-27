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

#ifndef rtkRayQuadricIntersectionFunction_h
#define rtkRayQuadricIntersectionFunction_h

#include <itkNumericTraits.h>
#include <vector>
#include <itkImageBase.h>

#include "rtkMacro.h"

namespace rtk
{

/** \class RayQuadricIntersectionFunction
 * \brief Test if a ray intersects with a Quadric.
 *
 * Return intersection points if there are some. See
 * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
 * for information on how this is implemented.
 *
 * \author Simon Rit
 *
 * \ingroup Functions
 */
template <
class TCoordRep = double,
unsigned int VDimension=3
>
class ITK_EXPORT RayQuadricIntersectionFunction :
    public itk::Object
{
public:
  /** Standard class typedefs. */
  typedef RayQuadricIntersectionFunction Self;
  typedef itk::Object                    Superclass;
  typedef itk::SmartPointer<Self>        Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  typedef typename itk::ImageBase<VDimension>::ConstPointer ImageBaseConstPointer;


  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Useful defines. */
  typedef itk::Vector<TCoordRep, VDimension> VectorType;

  /** Evaluate the intersection points */
  bool Evaluate( const VectorType& input );

  /** Get / Set the quadric parameters. */
  itkGetMacro(A, TCoordRep);
  itkSetMacro(A, TCoordRep);
  itkGetMacro(B, TCoordRep);
  itkSetMacro(B, TCoordRep);
  itkGetMacro(C, TCoordRep);
  itkSetMacro(C, TCoordRep);
  itkGetMacro(D, TCoordRep);
  itkSetMacro(D, TCoordRep);
  itkGetMacro(E, TCoordRep);
  itkSetMacro(E, TCoordRep);
  itkGetMacro(F, TCoordRep);
  itkSetMacro(F, TCoordRep);
  itkGetMacro(G, TCoordRep);
  itkSetMacro(G, TCoordRep);
  itkGetMacro(H, TCoordRep);
  itkSetMacro(H, TCoordRep);
  itkGetMacro(I, TCoordRep);
  itkSetMacro(I, TCoordRep);
  itkGetMacro(J, TCoordRep);
  itkSetMacro(J, TCoordRep);

  /** Get / Set the ray origin. */
  itkGetMacro(RayOrigin, VectorType);
  itkSetMacro(RayOrigin, VectorType);

  /** Get the distance with the nearest intersection.
    * \warning Only relevant if called after Evaluate. */
  itkGetMacro(NearestDistance, TCoordRep);

  /** Get the distance with the farthest intersection.
    * \warning Only relevant if called after Evaluate. */
  itkGetMacro(FarthestDistance, TCoordRep);


protected:

  /// Constructor
  RayQuadricIntersectionFunction();

  /// Destructor
  ~RayQuadricIntersectionFunction() {}

  /// The focal point or position of the ray source
  VectorType m_FocalPoint;

  /** Parameters of the quadratic shape */
  TCoordRep  m_A;
  TCoordRep  m_B;
  TCoordRep  m_C;
  TCoordRep  m_D;
  TCoordRep  m_E;
  TCoordRep  m_F;
  TCoordRep  m_G;
  TCoordRep  m_H;
  TCoordRep  m_I;
  TCoordRep  m_J;
  VectorType m_RayOrigin;
  VectorType m_RayDirection;
  TCoordRep  m_NearestDistance;
  TCoordRep  m_FarthestDistance;

private:
  RayQuadricIntersectionFunction( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented
};

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayQuadricIntersectionFunction.hxx"
#endif

#endif
