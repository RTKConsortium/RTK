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

#ifndef rtkRayConvexIntersectionImageFilter_h
#define rtkRayConvexIntersectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkConfiguration.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConvexShape.h"

namespace rtk
{

/** \class RayConvexIntersectionImageFilter
 * \brief Analytical projection of ConvexShape
 *
 * \test rtkfdktest.cxx, rtkforbildtest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class RayConvexIntersectionImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RayConvexIntersectionImageFilter                  Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Convenient typedefs. */
  typedef typename TOutputImage::RegionType     OutputImageRegionType;
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  typedef typename GeometryType::Pointer        GeometryPointer;
  typedef ConvexShape::Pointer                  ConvexShapePointer;
  typedef ConvexShape::ScalarType               ScalarType;
  typedef ConvexShape::PointType                PointType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayConvexIntersectionImageFilter, itk::InPlaceImageFilter);

  /** Get / Set the object pointer to the ConvexShape. */
  itkGetModifiableObjectMacro(ConvexShape, ConvexShape);
  itkSetObjectMacro(ConvexShape, ConvexShape);

  /** Get / Set the object pointer to projection geometry */
  itkGetModifiableObjectMacro(Geometry, GeometryType);
  itkSetObjectMacro(Geometry, GeometryType);

protected:
  RayConvexIntersectionImageFilter();
  ~RayConvexIntersectionImageFilter() {}

  /** ConvexShape must be created in the BeforeThreadedGenerateData in the
   * daugter classes. */
  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  /** Apply changes to the input image requested region. */
  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread,
                             ThreadIdType threadId ) ITK_OVERRIDE;

private:
  RayConvexIntersectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                   //purposely not implemented

  ConvexShapePointer m_ConvexShape;
  GeometryPointer    m_Geometry;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayConvexIntersectionImageFilter.hxx"
#endif

#endif
