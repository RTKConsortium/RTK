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

#ifndef rtkRayConvexObjectIntersectionImageFilter_h
#define rtkRayConvexObjectIntersectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkConfiguration.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConvexObject.h"

namespace rtk
{

/** \class RayConvexObjectIntersectionImageFilter
 * \brief Computes intersection of projection rays with convex objects.
 *
 * ConvexObjects are ellipsoid, cone, cylinder... See
 * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
 * for more information.
 *
 * \author Simon Rit
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class RayConvexObjectIntersectionImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RayConvexObjectIntersectionImageFilter            Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Convenient typedefs. */
  typedef typename TOutputImage::RegionType               OutputImageRegionType;
  typedef rtk::ThreeDCircularProjectionGeometry           GeometryType;
  typedef typename GeometryType::Pointer                  GeometryPointer;
  typedef ConvexObject::Pointer                           ConvexObjectPointer;
  typedef ConvexObject::ScalarType                        ScalarType;
  typedef ConvexObject::PointType                         PointType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayConvexObjectIntersectionImageFilter, itk::InPlaceImageFilter);

  /** Get / Set the object pointer to the geometry. */
  itkGetObjectMacro(ConvexObject, ConvexObject);
  itkSetObjectMacro(ConvexObject, ConvexObject);

  /** Get / Set the object pointer to projection geometry */
  itkGetObjectMacro(Geometry, GeometryType);
  itkSetObjectMacro(Geometry, GeometryType);

protected:
  RayConvexObjectIntersectionImageFilter();
  ~RayConvexObjectIntersectionImageFilter() {}

  /** ConvexObject must be created in the BeforeThreadedGenerateData in the
   * daugter classes. */
  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  /** Apply changes to the input image requested region. */
  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread,
                             ThreadIdType threadId ) ITK_OVERRIDE;

private:
  RayConvexObjectIntersectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                         //purposely not implemented

  ConvexObjectPointer     m_ConvexObject;

  GeometryPointer m_Geometry;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayConvexObjectIntersectionImageFilter.hxx"
#endif

#endif
