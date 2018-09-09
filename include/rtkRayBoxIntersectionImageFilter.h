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

#ifndef rtkRayBoxIntersectionImageFilter_h
#define rtkRayBoxIntersectionImageFilter_h

#include "rtkRayConvexIntersectionImageFilter.h"
#include "rtkConfiguration.h"
#include "rtkBoxShape.h"

namespace rtk
{

/** \class RayBoxIntersectionImageFilter
 * \brief Analytical projection of a BoxShape.
 *
 * \test rtkdrawgeometricphantomtest.cxx, rtkforbildtest.cxx
 *
 * \author Marc Vila, Simon Rit
 *
 * \ingroup RTK InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class RayBoxIntersectionImageFilter :
public RayConvexIntersectionImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef RayBoxIntersectionImageFilter                              Self;
  typedef RayConvexIntersectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                    Pointer;
  typedef itk::SmartPointer<const Self>                              ConstPointer;

  /** Convenient typedefs. */
  typedef BoxShape::PointType          PointType;
  typedef BoxShape::VectorType         VectorType;
  typedef BoxShape::ScalarType         ScalarType;
  typedef BoxShape::RotationMatrixType RotationMatrixType;
  typedef BoxShape::ImageBaseType      ImageBaseType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayBoxIntersectionImageFilter, RayConvexIntersectionImageFilter);

  /** Get / Set the constant density of the volume */
  itkGetMacro(Density, ScalarType);
  itkSetMacro(Density, ScalarType);

  /** Get reference to vector of plane parameters. */
  itkGetConstReferenceMacro(PlaneDirections, std::vector<VectorType>);
  itkGetConstReferenceMacro(PlanePositions, std::vector<ScalarType>);

  /** See ConvexShape for the definition of clip planes. */
  void AddClipPlane(const VectorType & dir, const ScalarType & pos);

  /** Set the box from an image. See rtk::BoxShape::SetBoxFromImage. */
  void SetBoxFromImage(const ImageBaseType * img, bool bWithExternalHalfPixelBorder=true);

  /** Get/Set the box parameters. See rtk::BoxShape. */
  itkGetMacro(BoxMin, PointType);
  itkSetMacro(BoxMin, PointType);
  itkGetMacro(BoxMax, PointType);
  itkSetMacro(BoxMax, PointType);
  itkGetMacro(Direction, RotationMatrixType);
  itkSetMacro(Direction, RotationMatrixType);

protected:
  RayBoxIntersectionImageFilter();
  ~RayBoxIntersectionImageFilter() ITK_OVERRIDE {}

  void BeforeThreadedGenerateData ( ) ITK_OVERRIDE;

private:
  RayBoxIntersectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

  ScalarType              m_Density;
  std::vector<VectorType> m_PlaneDirections;
  std::vector<ScalarType> m_PlanePositions;

  PointType               m_BoxMin;
  PointType               m_BoxMax;
  RotationMatrixType      m_Direction;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayBoxIntersectionImageFilter.hxx"
#endif

#endif
