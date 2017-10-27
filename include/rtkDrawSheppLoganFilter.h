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

#ifndef rtkDrawSheppLoganFilter_h
#define rtkDrawSheppLoganFilter_h

#include <itkInPlaceImageFilter.h>

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"

namespace rtk
{

/** \class DrawSheppLoganFilter
 * \brief Draws in a 3D image the Shepp-Logan phantom described in
 *  http://www.slaney.org/pct/pct-errata.html
 * Y and Z have been exchanged to follow the coordinate system of the IEC 61217
 * international standard used by RTK.
 *
 * \test rtkdrawgeometricphantomtest.cxx,
 * rtkRaycastInterpolatorForwardProjectionTest.cxx,
 * rtkfdktest.cxx, rtkrampfiltertest.cxx, rtkforwardprojectiontest.cxx,
 * rtkdisplaceddetectortest.cxx, rtkshortscantest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT DrawSheppLoganFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef DrawSheppLoganFilter                              Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  typedef typename TOutputImage::RegionType                 OutputImageRegionType;

  typedef itk::Vector<double,3>                             VectorType;
  typedef std::string                                       StringType;

  typedef rtk::ConvertEllipsoidToQuadricParametersFunction  EQPFunctionType;
  struct FigureType
  {
    FigureType():angle(0.),attenuation(0.){};
    VectorType semiprincipalaxis;
    VectorType center;
    double     angle;
    double     attenuation;
  };
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawSheppLoganFilter, InPlaceImageFilter);

  /** Multiplicative Scaling factor for the phantom parameters described in
   * http://www.slaney.org/pct/pct-errata.html. */
  itkSetMacro(PhantomScale, VectorType);
  itkGetMacro(PhantomScale, VectorType);

  /** Get / Set the spatial position of the Shepp Logan phantom relative to its
   * center. The default value is (0, 0, 0). */
  itkSetMacro(OriginOffset, VectorType);
  itkGetMacro(OriginOffset, VectorType);

protected:
  DrawSheppLoganFilter();
  ~DrawSheppLoganFilter() {}

  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;
  void SetEllipsoid(FigureType* rei,
                    double spax,
                    double spay,
                    double spaz,
                    double centerx,
                    double centery,
                    double centerz,
                    double angle,
                    double density);
private:
  DrawSheppLoganFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  VectorType m_PhantomScale;
  VectorType m_OriginOffset;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawSheppLoganFilter.hxx"
#endif

#endif
