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

#ifndef rtkDrawQuadricImageFilter_h
#define rtkDrawQuadricImageFilter_h

#include "rtkDrawConvexImageFilter.h"
#include "rtkConfiguration.h"

namespace rtk
{

/** \class DrawQuadricImageFilter
 * \brief Draws in a 3D image user defined Quadric.
 *
 * \test rtkdrawgeometricphantomtest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class DrawQuadricImageFilter:
public DrawConvexImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef DrawQuadricImageFilter                          Self;
  typedef DrawConvexImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                         Pointer;
  typedef itk::SmartPointer<const Self>                   ConstPointer;

  /** Convenient typedefs. */
  typedef ConvexShape::VectorType VectorType;
  typedef ConvexShape::ScalarType ScalarType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawQuadricImageFilter, DrawConvexImageFilter);

  /** Get / Set the constant density of the volume */
  itkGetMacro(Density, ScalarType);
  itkSetMacro(Density, ScalarType);

  /** Get reference to vector of plane parameters. */
  itkGetConstReferenceMacro(PlaneDirections, std::vector<VectorType>);
  itkGetConstReferenceMacro(PlanePositions, std::vector<ScalarType>);

  void AddClipPlane(const VectorType & dir, const ScalarType & pos);

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
  DrawQuadricImageFilter();
  ~DrawQuadricImageFilter() {}

  void BeforeThreadedGenerateData ( ) ITK_OVERRIDE;

private:
  DrawQuadricImageFilter(const Self&); //purposely not implemented
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
#include "rtkDrawQuadricImageFilter.hxx"
#endif

#endif
