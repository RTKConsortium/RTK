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
class DrawBoxImageFilter :
public DrawConvexImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef DrawBoxImageFilter                              Self;
  typedef DrawConvexImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                         Pointer;
  typedef itk::SmartPointer<const Self>                   ConstPointer;

  /** Convenient typedefs. */
  typedef BoxShape::PointType          PointType;
  typedef BoxShape::VectorType         VectorType;
  typedef BoxShape::ScalarType         ScalarType;
  typedef BoxShape::RotationMatrixType RotationMatrixType;
  typedef BoxShape::ImageBaseType      ImageBaseType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawBoxImageFilter, DrawConvexImageFilter);

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
  itkGetMacro(BoxMin, VectorType);
  itkSetMacro(BoxMin, VectorType);
  itkGetMacro(BoxMax, VectorType);
  itkSetMacro(BoxMax, VectorType);
  itkGetMacro(Direction, RotationMatrixType);
  itkSetMacro(Direction, RotationMatrixType);

protected:
  DrawBoxImageFilter();
  ~DrawBoxImageFilter() {}

  void BeforeThreadedGenerateData ( ) ITK_OVERRIDE;

private:
  DrawBoxImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);     //purposely not implemented

  ScalarType              m_Density;
  std::vector<VectorType> m_PlaneDirections;
  std::vector<ScalarType> m_PlanePositions;

  VectorType         m_BoxMin;
  VectorType         m_BoxMax;
  RotationMatrixType m_Direction;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawBoxImageFilter.hxx"
#endif

#endif
