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

#ifndef rtkRayQuadricIntersectionImageFilter_h
#define rtkRayQuadricIntersectionImageFilter_h

#include "rtkRayConvexIntersectionImageFilter.h"
#include "rtkConfiguration.h"

namespace rtk
{

/** \class RayQuadricIntersectionImageFilter
 * \brief Analytical projection of a QuadricShape
 *
 * \test rtkdrawgeometricphantomtest.cxx, rtkforbildtest.cxx
 *
 * \author Marc Vila, Simon Rit
 *
 * \ingroup RTK InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class RayQuadricIntersectionImageFilter :
public RayConvexIntersectionImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef RayQuadricIntersectionImageFilter                          Self;
  typedef RayConvexIntersectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                    Pointer;
  typedef itk::SmartPointer<const Self>                              ConstPointer;

  /** Convenient typedefs. */
  typedef ConvexShape::PointType  PointType;
  typedef ConvexShape::VectorType VectorType;
  typedef ConvexShape::ScalarType ScalarType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayQuadricIntersectionImageFilter, RayConvexIntersectionImageFilter);

  /** Get / Set the constant density of the volume */
  itkGetMacro(Density, ScalarType);
  itkSetMacro(Density, ScalarType);

  /** Get reference to vector of plane parameters. */
  itkGetConstReferenceMacro(PlaneDirections, std::vector<VectorType>);
  itkGetConstReferenceMacro(PlanePositions, std::vector<ScalarType>);

  /** See ConvexShape for the definition of clip planes. */
  void AddClipPlane(const VectorType & dir, const ScalarType & pos);

  /** Get/Set QuadricShape parameters. */
  itkGetMacro(A, ScalarType);
  itkSetMacro(A, ScalarType);
  itkGetMacro(B, ScalarType);
  itkSetMacro(B, ScalarType);
  itkGetMacro(C, ScalarType);
  itkSetMacro(C, ScalarType);
  itkGetMacro(D, ScalarType);
  itkSetMacro(D, ScalarType);
  itkGetMacro(E, ScalarType);
  itkSetMacro(E, ScalarType);
  itkGetMacro(F, ScalarType);
  itkSetMacro(F, ScalarType);
  itkGetMacro(G, ScalarType);
  itkSetMacro(G, ScalarType);
  itkGetMacro(H, ScalarType);
  itkSetMacro(H, ScalarType);
  itkGetMacro(I, ScalarType);
  itkSetMacro(I, ScalarType);
  itkGetMacro(J, ScalarType);
  itkSetMacro(J, ScalarType);

protected:
  RayQuadricIntersectionImageFilter();
  virtual ~RayQuadricIntersectionImageFilter() ITK_OVERRIDE {}

  void BeforeThreadedGenerateData ( ) ITK_OVERRIDE;

private:
  RayQuadricIntersectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

  ScalarType              m_Density;
  std::vector<VectorType> m_PlaneDirections;
  std::vector<ScalarType> m_PlanePositions;

  ScalarType m_A;
  ScalarType m_B;
  ScalarType m_C;
  ScalarType m_D;
  ScalarType m_E;
  ScalarType m_F;
  ScalarType m_G;
  ScalarType m_H;
  ScalarType m_I;
  ScalarType m_J;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayQuadricIntersectionImageFilter.hxx"
#endif

#endif
