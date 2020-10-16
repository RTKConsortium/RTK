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

#ifndef rtkDrawEllipsoidImageFilter_h
#define rtkDrawEllipsoidImageFilter_h

#include "rtkDrawConvexImageFilter.h"
#include "rtkConfiguration.h"

namespace rtk
{

/** \class DrawEllipsoidImageFilter
 * \brief Draws an ellipsoid in a 3D image
 *
 * \test rtkdrawgeometricphantomtest.cxx, rtkforbildtest.cxx
 *
 * \author Marc Vila, Simon Rit
 *
 * \ingroup RTK InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class DrawEllipsoidImageFilter : public DrawConvexImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(DrawEllipsoidImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(DrawEllipsoidImageFilter);
#endif

  /** Standard class type alias. */
  using Self = DrawEllipsoidImageFilter;
  using Superclass = DrawConvexImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  using PointType = ConvexShape::PointType;
  using VectorType = ConvexShape::VectorType;
  using ScalarType = ConvexShape::ScalarType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawEllipsoidImageFilter, DrawConvexImageFilter);

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
  DrawEllipsoidImageFilter();
  ~DrawEllipsoidImageFilter() override = default;

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
#  include "rtkDrawEllipsoidImageFilter.hxx"
#endif

#endif
