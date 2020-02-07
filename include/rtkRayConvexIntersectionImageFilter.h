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
 * \ingroup RTK InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class RayConvexIntersectionImageFilter : public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(RayConvexIntersectionImageFilter);

  /** Standard class type alias. */
  using Self = RayConvexIntersectionImageFilter;
  using Superclass = itk::InPlaceImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  using GeometryConstPointer = typename GeometryType::ConstPointer;
  using ConvexShapePointer = ConvexShape::Pointer;
  using ScalarType = ConvexShape::ScalarType;
  using PointType = ConvexShape::PointType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayConvexIntersectionImageFilter, itk::InPlaceImageFilter);

  /** Get / Set the object pointer to the ConvexShape. */
  itkGetModifiableObjectMacro(ConvexShape, ConvexShape);
  itkSetObjectMacro(ConvexShape, ConvexShape);

  /** Get / Set the object pointer to projection geometry */
  itkGetConstObjectMacro(Geometry, GeometryType);
  itkSetConstObjectMacro(Geometry, GeometryType);

  /** Get / Set the attenuation \f$\mu\f$ to simulate the attenuated line integral. Default is 0.
   * The attenuated line integral model is
   * \f[
   * p=\int f(\underline s+r\underline{d})\exp(\mu r)\mathrm{d}r
   * \f]
   * with \f$\underline s\f$ and \f$\underline d\f$ the ray source and direction. The value is therefore:
   * \f{eqnarray*}{
   * p&=&\int_{n}^{f} d\exp(\mu r)\mathrm{d}r\\
   * &=&\dfrac{d}{\mu}\left(\exp(\mu {f})-\exp(\mu n)\right)
   * \f}
   * with \f$n\f$ and \f$f\f$ the distances from source to intersection points near and far from the source and \f$d\f$
   * m_Density.
   */
  itkGetMacro(Attenuation, double);
  itkSetMacro(Attenuation, double);

protected:
  RayConvexIntersectionImageFilter();
  ~RayConvexIntersectionImageFilter() override = default;

  /** ConvexShape must be created in the BeforeThreadedGenerateData in the
   * daugter classes. */
  void
  BeforeThreadedGenerateData() override;

  /** Apply changes to the input image requested region. */
  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

private:
  ConvexShapePointer   m_ConvexShape;
  GeometryConstPointer m_Geometry;
  double               m_Attenuation = 0.;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkRayConvexIntersectionImageFilter.hxx"
#endif

#endif
