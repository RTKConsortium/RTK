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
class RayQuadricIntersectionImageFilter : public RayConvexIntersectionImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(RayQuadricIntersectionImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(RayQuadricIntersectionImageFilter);
#endif

  /** Standard class type alias. */
  using Self = RayQuadricIntersectionImageFilter;
  using Superclass = RayConvexIntersectionImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  using PointType = ConvexShape::PointType;
  using VectorType = ConvexShape::VectorType;
  using ScalarType = ConvexShape::ScalarType;

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
  void
  AddClipPlane(const VectorType & dir, const ScalarType & pos);

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
  ~RayQuadricIntersectionImageFilter() override = default;

  void
  BeforeThreadedGenerateData() override;

private:
  ScalarType              m_Density{ 1. };
  std::vector<VectorType> m_PlaneDirections;
  std::vector<ScalarType> m_PlanePositions;

  ScalarType m_A{ 0. };
  ScalarType m_B{ 0. };
  ScalarType m_C{ 0. };
  ScalarType m_D{ 0. };
  ScalarType m_E{ 0. };
  ScalarType m_F{ 0. };
  ScalarType m_G{ 0. };
  ScalarType m_H{ 0. };
  ScalarType m_I{ 0. };
  ScalarType m_J{ 0. };
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkRayQuadricIntersectionImageFilter.hxx"
#endif

#endif
