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

#include <itkInPlaceImageFilter.h>
#include "rtkConfiguration.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkRayQuadricIntersectionFunction.h"

namespace rtk
{

/** \class RayQuadricIntersectionImageFilter
 * \brief Computes intersection of projection rays with quadric objects.
 *
 * Quadrics are ellipsoid, cone, cylinder... See
 * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
 * for more information.
 *
 * \author Simon Rit
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT RayQuadricIntersectionImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RayQuadricIntersectionImageFilter                 Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  typedef typename TOutputImage::RegionType               OutputImageRegionType;
  typedef typename TOutputImage::Superclass::ConstPointer OutputImageBaseConstPointer;
  typedef rtk::ThreeDCircularProjectionGeometry           GeometryType;
  typedef typename GeometryType::Pointer                  GeometryPointer;
  typedef RayQuadricIntersectionFunction<double, 3>       RQIFunctionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayQuadricIntersectionImageFilter, itk::ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

  /** Get / Set the multiplicative constant of the volume */
  itkGetMacro(Density, double);
  itkSetMacro(Density, double);

  /** Get the RayQuadricIntersectionFunction to set its parameters.
    * A call to this function will assume modification of the function.*/
  RQIFunctionType::Pointer GetRQIFunctor();

protected:
  RayQuadricIntersectionImageFilter();
  ~RayQuadricIntersectionImageFilter() {}

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  /** Apply changes to the input image requested region. */
  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread,
                                     ThreadIdType threadId ) ITK_OVERRIDE;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

private:
  RayQuadricIntersectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  /** Functor object to compute the intersection */
  RQIFunctionType::Pointer m_RQIFunctor;

  /** RTK geometry object */
  GeometryPointer m_Geometry;

  /** Multiplicative factor of intersection length */
  double m_Density;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayQuadricIntersectionImageFilter.hxx"
#endif

#endif
