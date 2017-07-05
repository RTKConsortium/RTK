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

#ifndef rtkSheppLoganPhantomFilter_h
#define rtkSheppLoganPhantomFilter_h

#include "rtkRayEllipsoidIntersectionImageFilter.h"

namespace rtk
{

/** \class SheppLoganPhantomFilter
 * \brief Computes intersection between source rays and ellipsoids,
 * in order to create the projections of a Shepp-Logan phantom resized
 * to m_PhantoScale ( default 128 ).
 *
 * \test rtkRaycastInterpolatorForwardProjectionTest.cxx,
 * rtkprojectgeometricphantomtest.cxx, rtkfdktest.cxx, rtkrampfiltertest.cxx,
 * rtkforwardprojectiontest.cxx, rtkdisplaceddetectortest.cxx,
 * rtkshortscantest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT SheppLoganPhantomFilter:
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef SheppLoganPhantomFilter                                  Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage>        Superclass;
  typedef itk::SmartPointer<Self>                                  Pointer;
  typedef itk::SmartPointer<const Self>                            ConstPointer;

  typedef rtk::RayEllipsoidIntersectionImageFilter< TInputImage,
                                                    TOutputImage > REIType;
  typedef itk::Vector<double,3>                                    VectorType;
  typedef std::string                                              StringType;
  typedef std::vector< std::vector<double> >                       VectorOfVectorType;
  typedef rtk::ThreeDCircularProjectionGeometry                    GeometryType;
  typedef typename GeometryType::Pointer                           GeometryPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SheppLoganPhantomFilter, itk::InPlaceImageFilter);

  /** Get / Set the scaling factor of the spatial dimensions of the phantom. By
   * default, the scaling factor is 128 and the outer ellipse of the phantom is
   * 88.32 x 115.2 x 117.76. */
  itkSetMacro(PhantomScale, VectorType);
  itkGetMacro(PhantomScale, VectorType);

  /** Get / Set the spatial position of the Shepp Logan phantom relative to its`
   * center. The default value is (0, 0, 0). The offset is applied before
   * PhantomScale.*/
  itkSetMacro(OriginOffset, VectorType);
  itkGetMacro(OriginOffset, VectorType);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

protected:
  SheppLoganPhantomFilter();
  ~SheppLoganPhantomFilter() {}

  void GenerateData() ITK_OVERRIDE;
  void SetEllipsoid(typename REIType::Pointer rei,
                    double spax,
                    double spay,
                    double spaz,
                    double centerx,
                    double centery,
                    double centerz,
                    double angle,
                    double density);
private:
  SheppLoganPhantomFilter(const Self&); //purposely not implemented
  void operator=(const Self&);          //purposely not implemented

  VectorType      m_PhantomScale;
  VectorType      m_OriginOffset;
  GeometryPointer m_Geometry;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSheppLoganPhantomFilter.hxx"
#endif

#endif
