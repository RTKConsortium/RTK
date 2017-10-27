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

#ifndef rtkCudaRayCastBackProjectionImageFilter_h
#define rtkCudaRayCastBackProjectionImageFilter_h

#include "rtkBackProjectionImageFilter.h"
#include "rtkWin32Header.h"

#include <itkCudaImage.h>
#include <itkCudaInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

/** \class CudaRayCastBackProjectionImageFilter
 * \brief Cuda version of the  backprojection.
 *
 * GPU-based implementation of the backprojection step of the
 * [Feldkamp, Davis, Kress, 1984] algorithm for filtered backprojection
 * reconstruction of cone-beam CT images with a circular source trajectory.
 *
 * \test rtksarttest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup Projector CudaImageToImageFilter
 */
class CudaRayCastBackProjectionImageFilter :
  public itk::CudaInPlaceImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
  BackProjectionImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3> > >
{
public:
  /** Standard class typedefs. */
  typedef itk::CudaImage<float,3>                           ImageType;
  typedef BackProjectionImageFilter< ImageType, ImageType>  BackProjectionImageFilterType;
  typedef CudaRayCastBackProjectionImageFilter              Self;
  typedef itk::CudaInPlaceImageFilter<ImageType, ImageType,
                     BackProjectionImageFilterType>         Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  typedef ImageType::RegionType                             OutputImageRegionType;
  typedef itk::CudaImage<float, 2>                          ProjectionImageType;
  typedef ProjectionImageType::Pointer                      ProjectionImagePointer;
  typedef rtk::ThreeDCircularProjectionGeometry             GeometryType;
  typedef GeometryType::Pointer                             GeometryPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaRayCastBackProjectionImageFilter, Superclass);

  /** Set step size along ray (in mm). Default is 1 mm. */
  itkGetConstMacro(StepSize, double);
  itkSetMacro(StepSize, double);

  /** Set whether the back projection should be divided by the sum of splat weights */
  itkGetMacro(Normalize, bool);
  itkSetMacro(Normalize, bool);

protected:
  rtkcuda_EXPORT CudaRayCastBackProjectionImageFilter();
  virtual ~CudaRayCastBackProjectionImageFilter() {};

  virtual void GPUGenerateData();

private:
  CudaRayCastBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                   //purposely not implemented

  double             m_StepSize;
  bool               m_Normalize;
};

} // end namespace rtk

#endif
