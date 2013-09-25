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

#ifndef __rtkDrawEllipsoidImageFilter_h
#define __rtkDrawEllipsoidImageFilter_h

#include <itkInPlaceImageFilter.h>

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"
#include "rtkConfiguration.h"
#include <vector>

namespace rtk
{

/** \class DrawEllipsoidImageFilter
 * \brief Draws in a 3D image a user defined ellipsoid.
 *
 * \test rtksarttest.cxx, rtkmotioncompensatedfdktest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT DrawEllipsoidImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef DrawEllipsoidImageFilter                          Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  typedef typename TOutputImage::RegionType                 OutputImageRegionType;

  typedef itk::Vector<double,3>                             VectorType;

  typedef rtk::ConvertEllipsoidToQuadricParametersFunction  EQPFunctionType;
  struct FigureType
  {
    FigureType():angle(0.),density(0.){};
    VectorType semiprincipalaxis;
    VectorType center;
    double     angle;
    double     density;
  };
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawEllipsoidImageFilter, InPlaceImageFilter);

  /** Multiplicative Scaling factor for the phantom parameters described in
   * http://www.slaney.org/pct/pct-errata.html. */
  itkSetMacro(Density, double);
  itkGetMacro(Density, double);

  itkSetMacro(Angle, double);
  itkGetMacro(Angle, double);

  itkSetMacro(Axis, VectorType);
  itkGetMacro(Axis, VectorType);

  itkSetMacro(Center, VectorType);
  itkGetMacro(Center, VectorType);

protected:
  DrawEllipsoidImageFilter();
  virtual ~DrawEllipsoidImageFilter() {};

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

private:
  DrawEllipsoidImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);           //purposely not implemented

  VectorType      m_Axis;
  VectorType      m_Center;
  double          m_Density;
  double          m_Angle;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawEllipsoidImageFilter.txx"
#endif

#endif
