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
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class DrawEllipsoidImageFilter :
public DrawConvexImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef DrawEllipsoidImageFilter                        Self;
  typedef DrawConvexImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                         Pointer;
  typedef itk::SmartPointer<const Self>                   ConstPointer;

  /** Convenient typedefs. */
  typedef ConvexShape::PointType  PointType;
  typedef ConvexShape::VectorType VectorType;
  typedef ConvexShape::ScalarType ScalarType;

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
  void AddClipPlane(const VectorType & dir, const ScalarType & pos);

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
  virtual ~DrawEllipsoidImageFilter() ITK_OVERRIDE {}

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

private:
  DrawEllipsoidImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

  ScalarType              m_Density;
  std::vector<VectorType> m_PlaneDirections;
  std::vector<ScalarType> m_PlanePositions;

  PointType  m_Center;
  VectorType m_Axis;
  ScalarType m_Angle;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawEllipsoidImageFilter.hxx"
#endif

#endif
