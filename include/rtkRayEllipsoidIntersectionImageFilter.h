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

#ifndef rtkRayEllipsoidIntersectionImageFilter_h
#define rtkRayEllipsoidIntersectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkRayQuadricIntersectionImageFilter.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"

namespace rtk
{

/** \class RayEllipsoidIntersectionImageFilter
 * \brief Analytical projection of ellipsoids
 *
 * \test rtksarttest.cxx, rtkamsterdamshroudtest.cxx,
 *        rtkmotioncompensatedfdktest.cxx, rtkforbildtest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT RayEllipsoidIntersectionImageFilter
  : public RayConvexIntersectionImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(RayEllipsoidIntersectionImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(RayEllipsoidIntersectionImageFilter);
#endif

  /** Standard class type alias. */
  using Self = RayEllipsoidIntersectionImageFilter;
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
  itkTypeMacro(RayEllipsoidIntersectionImageFilter, RayConvexIntersectionImageFilter);

  /** Get / Set the constant density of the volume */
  itkGetMacro(Density, ScalarType);
  itkSetMacro(Density, ScalarType);

  /** Get reference to vector of plane parameters. */
  itkGetConstReferenceMacro(PlaneDirections, std::vector<VectorType>);
  itkGetConstReferenceMacro(PlanePositions, std::vector<ScalarType>);

  /** See ConvexShape for the definition of clip planes. */
  void
  AddClipPlane(const VectorType & dir, const ScalarType & pos);

  /** Get/Set the center of the ellipsoid. */
  itkGetMacro(Center, PointType);
  itkSetMacro(Center, PointType);

  /** Get/Set the semi-principal axes of the ellipsoid. */
  itkGetMacro(Axis, VectorType);
  itkSetMacro(Axis, VectorType);

  /** Get/Set the rotation angle around the y axis. */
  itkGetMacro(Angle, ScalarType);
  itkSetMacro(Angle, ScalarType);

protected:
  RayEllipsoidIntersectionImageFilter();
  ~RayEllipsoidIntersectionImageFilter() override = default;

  void
  BeforeThreadedGenerateData() override;

private:
  ScalarType              m_Density{ 1. };
  std::vector<VectorType> m_PlaneDirections;
  std::vector<ScalarType> m_PlanePositions;

  PointType  m_Center;
  VectorType m_Axis;
  ScalarType m_Angle{ 0. };
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkRayEllipsoidIntersectionImageFilter.hxx"
#endif

#endif
