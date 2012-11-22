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

#ifndef __rtkSheppLoganPhantomFilter_h
#define __rtkSheppLoganPhantomFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkRayQuadricIntersectionImageFilter.h"

#include "rtkRayEllipsoidIntersectionImageFilter.h"

namespace rtk
{

/** \class SheppLoganPhantomFilter
 * \brief Computes intersection between source rays and ellipsoids,
 * in order to create the projections of a Shepp-Logan phantom resized
 * to m_PhantoScale ( default 128 ).
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT SheppLoganPhantomFilter :
  public RayEllipsoidIntersectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef SheppLoganPhantomFilter                                       Self;
  typedef RayEllipsoidIntersectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                       Pointer;
  typedef itk::SmartPointer<const Self>                                 ConstPointer;
  typedef typename TOutputImage::RegionType               OutputImageRegionType;
  typedef typename TOutputImage::Superclass::ConstPointer OutputImageBaseConstPointer;

  typedef float OutputPixelType;

  typedef itk::Image< OutputPixelType, 3 >                                           OutputImageType;
  typedef rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType> REIType;
  typedef std::vector<double>                                                        VectorType;
  typedef std::string                                                                StringType;
  typedef std::vector< std::vector<double> >                                         VectorOfVectorType;
  struct FigureType
  {
    FigureType():angle(0.),attenuation(0.){};
    VectorType semiprincipalaxis;
    VectorType center;
    double angle;
    double attenuation;
    };
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SheppLoganPhantomFilter, RayEllipsoidIntersectionImageFilter);

  itkSetMacro(PhantomScale, double);
  itkGetMacro(PhantomScale, double);
  itkSetMacro(PhantomOriginOffsetX, double);
  itkGetMacro(PhantomOriginOffsetX, double);

protected:
  SheppLoganPhantomFilter();
  virtual ~SheppLoganPhantomFilter() {};

  virtual void GenerateData();

  /** Translate user parameteres to quadric parameters.
   * A call to this function will assume modification of the function.*/


private:
  SheppLoganPhantomFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  double                   m_PhantomScale;
  double                   m_PhantomOriginOffsetX;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSheppLoganPhantomFilter.txx"
#endif

#endif
