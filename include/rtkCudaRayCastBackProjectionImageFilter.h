/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkBackProjectionImageFilter.h"
#  include "RTKExport.h"

#  include <itkCudaImage.h>
#  include <itkCudaInPlaceImageFilter.h>
#  include "rtkThreeDCircularProjectionGeometry.h"

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
 * \ingroup RTK Projector CudaImageToImageFilter
 */
class RTK_EXPORT CudaRayCastBackProjectionImageFilter
  : public itk::CudaInPlaceImageFilter<itk::CudaImage<float, 3>,
                                       itk::CudaImage<float, 3>,
                                       BackProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaRayCastBackProjectionImageFilter);

  /** Standard class type alias. */
  using ImageType = itk::CudaImage<float, 3>;
  using BackProjectionImageFilterType = BackProjectionImageFilter<ImageType, ImageType>;
  using Self = CudaRayCastBackProjectionImageFilter;
  using Superclass = itk::CudaInPlaceImageFilter<ImageType, ImageType, BackProjectionImageFilterType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using OutputImageRegionType = ImageType::RegionType;
  using ProjectionImageType = itk::CudaImage<float, 2>;
  using ProjectionImagePointer = ProjectionImageType::Pointer;
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  using GeometryPointer = GeometryType::Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(CudaRayCastBackProjectionImageFilter);

  /** Set step size along ray (in mm). Default is 1 mm. */
  itkGetConstMacro(StepSize, double);
  itkSetMacro(StepSize, double);

  /** Set whether the back projection should be divided by the sum of splat weights */
  itkGetMacro(Normalize, bool);
  itkSetMacro(Normalize, bool);

protected:
  CudaRayCastBackProjectionImageFilter();
  virtual ~CudaRayCastBackProjectionImageFilter(){};

  virtual void
  GPUGenerateData();

private:
  double m_StepSize;
  bool   m_Normalize;
};

} // end namespace rtk

#endif // end conditional definition of the class

#endif
