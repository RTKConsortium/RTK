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

#ifndef rtkRayBoxIntersectionImageFilter_h
#define rtkRayBoxIntersectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkRayBoxIntersectionFunction.h"
#include "rtkConfiguration.h"

namespace rtk
{

/** \class RayBoxIntersectionImageFilter
 * \brief Computes intersection of projection rays with image box.
 *
 * The filter uses RayBoxIntersectionFunction.
 *
 * \test rtkRaycastInterpolatorForwardProjectionTest.cxx,
 * rtkforwardprojectiontest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup InPlaceImageFilter
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT RayBoxIntersectionImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RayBoxIntersectionImageFilter                     Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);

  typedef typename TOutputImage::RegionType                                          OutputImageRegionType;
  typedef rtk::ThreeDCircularProjectionGeometry                                      GeometryType;
  typedef typename GeometryType::Pointer                                             GeometryPointer;
  typedef RayBoxIntersectionFunction<double, itkGetStaticConstMacro(ImageDimension)> RBIFunctionType;
  typedef itk::ImageBase<itkGetStaticConstMacro(ImageDimension)>                     ImageBaseType;

  /** Useful defines. */
  typedef itk::Vector<double, 3> VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayBoxIntersectionImageFilter, itk::ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

  /** Set the box from an image. See rtkRayBoxIntersectionFunction::SetBoxFromImage. */
  void SetBoxFromImage(const ImageBaseType * img, bool bWithExternalHalfPixelBorder=true);

  /** Set the box from an image */
  void SetBoxMin(VectorType _boxMin);

  /** Set the box from an image */
  void SetBoxMax(VectorType _boxMax);

  /** Get / Set the multiplicative constant of the volume */
  itkGetMacro(Density, double);
  itkSetMacro(Density, double);


protected:
  RayBoxIntersectionImageFilter() : m_RBIFunctor(RBIFunctionType::New()), m_Geometry(ITK_NULLPTR), m_Density(1.) {}
  virtual ~RayBoxIntersectionImageFilter() ITK_OVERRIDE {}

  /** Apply changes to the input image requested region. */
  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

private:
  RayBoxIntersectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  /** Functor object to compute the intersection */
  typename RBIFunctionType::Pointer m_RBIFunctor;

  /** RTK geometry object */
  GeometryPointer m_Geometry;

  double m_Density;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayBoxIntersectionImageFilter.hxx"
#endif

#endif
