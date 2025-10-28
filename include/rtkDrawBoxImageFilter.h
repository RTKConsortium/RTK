/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkDrawBoxImageFilter_h
#define rtkDrawBoxImageFilter_h

#include "rtkDrawConvexImageFilter.h"
#include "rtkConfiguration.h"
#include "rtkBoxShape.h"

namespace rtk
{

/** \class DrawBoxImageFilter
 * \brief Draws a 3D image user defined BoxShape.
 *
 * \test rtkdrawgeometricphantomtest.cxx, rtkforbildtest.cxx
 *
 * \author Marc Vila, Simon Rit
 *
 * \ingroup RTK InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT DrawBoxImageFilter : public DrawConvexImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DrawBoxImageFilter);

  /** Standard class type alias. */
  using Self = DrawBoxImageFilter;
  using Superclass = DrawConvexImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  using PointType = BoxShape::PointType;
  using VectorType = BoxShape::VectorType;
  using ScalarType = BoxShape::ScalarType;
  using RotationMatrixType = BoxShape::RotationMatrixType;
  using ImageBaseType = BoxShape::ImageBaseType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(DrawBoxImageFilter);

  /** Get / Set the constant density of the volume */
  itkGetMacro(Density, ScalarType);
  itkSetMacro(Density, ScalarType);

  /** Get reference to vector of plane parameters. */
  itkGetConstReferenceMacro(PlaneDirections, std::vector<VectorType>);
  itkGetConstReferenceMacro(PlanePositions, std::vector<ScalarType>);

  /** See ConvexShape for the definition of clip planes. */
  void
  AddClipPlane(const VectorType & dir, const ScalarType & pos);

  /** Set the box from an image. See rtk::BoxShape::SetBoxFromImage. */
  void
  SetBoxFromImage(const ImageBaseType * img, bool bWithExternalHalfPixelBorder = true);

  /** Get/Set the box parameters. See rtk::BoxShape. */
  itkGetMacro(BoxMin, PointType);
  itkSetMacro(BoxMin, PointType);
  itkGetMacro(BoxMax, PointType);
  itkSetMacro(BoxMax, PointType);
  itkGetMacro(Direction, RotationMatrixType);
  itkSetMacro(Direction, RotationMatrixType);

protected:
  DrawBoxImageFilter();
  ~DrawBoxImageFilter() = default;

  void
  BeforeThreadedGenerateData() override;

private:
  ScalarType              m_Density{ 1. };
  std::vector<VectorType> m_PlaneDirections;
  std::vector<ScalarType> m_PlanePositions;

  PointType          m_BoxMin{ 0. };
  PointType          m_BoxMax{ 0. };
  RotationMatrixType m_Direction;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkDrawBoxImageFilter.hxx"
#endif

#endif
