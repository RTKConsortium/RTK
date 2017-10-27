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

#ifndef rtkRayEllipsoidIntersectionImageFilter_h
#define rtkRayEllipsoidIntersectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkRayQuadricIntersectionImageFilter.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"

namespace rtk
{

/** \class RayEllipsoidIntersectionImageFilter
 * \brief Computes intersection of projection rays with ellipsoids.
 *
 * See http://en.wikipedia.org/wiki/Ellipsoid for more information.
 *
 * \test rtksarttest.cxx, rtkamsterdamshroudtest.cxx, rtkmotioncompensatedfdktest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT RayEllipsoidIntersectionImageFilter :
  public RayQuadricIntersectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RayEllipsoidIntersectionImageFilter                         Self;
  typedef RayQuadricIntersectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                     Pointer;
  typedef itk::SmartPointer<const Self>                               ConstPointer;

  typedef itk::Vector<double,3>                                       VectorType;

  typedef ConvertEllipsoidToQuadricParametersFunction                 EQPFunctionType;
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayEllipsoidIntersectionImageFilter, RayQuadricIntersectionImageFilter);

  /** Get/Set the semi-principal axes of the ellipsoid.*/

  itkSetMacro(Angle, double);
  itkGetMacro(Angle, double);

  itkSetMacro(Axis, VectorType);
  itkGetMacro(Axis, VectorType);

  itkSetMacro(Center, VectorType);
  itkGetMacro(Center, VectorType);

  itkSetMacro(Figure, std::string);
  itkGetMacro(Figure, std::string);

protected:
  RayEllipsoidIntersectionImageFilter();
  ~RayEllipsoidIntersectionImageFilter() {}

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

private:
  RayEllipsoidIntersectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  VectorType               m_Axis;
  VectorType               m_Center;
  double                   m_Attenuation;
  double                   m_Angle;
  std::string              m_Figure;
  EQPFunctionType::Pointer m_EQPFunctor;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayEllipsoidIntersectionImageFilter.hxx"
#endif

#endif
