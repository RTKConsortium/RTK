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

#ifndef rtkCudaBackProjectionImageFilter_h
#define rtkCudaBackProjectionImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkBackProjectionImageFilter.h"
#  include "RTKExport.h"

#  include <itkCudaImage.h>
#  include <itkCudaInPlaceImageFilter.h>

namespace rtk
{

/** \class CudaBackProjectionImageFilter
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

template <class ImageType = itk::CudaImage<float, 3>>
class ITK_TEMPLATE_EXPORT CudaBackProjectionImageFilter
  : public itk::CudaInPlaceImageFilter<ImageType, ImageType, BackProjectionImageFilter<ImageType, ImageType>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaBackProjectionImageFilter);

  /** Standard class type alias. */
  using BackProjectionImageFilterType = BackProjectionImageFilter<ImageType, ImageType>;
  using Self = CudaBackProjectionImageFilter;
  using Superclass = itk::CudaInPlaceImageFilter<ImageType, ImageType, BackProjectionImageFilterType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using OutputImageRegionType = typename ImageType::RegionType;
  using ProjectionImageType = itk::CudaImage<float, 2>;
  using ProjectionImagePointer = ProjectionImageType::Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(CudaBackProjectionImageFilter);

protected:
  CudaBackProjectionImageFilter();
  virtual ~CudaBackProjectionImageFilter(){};

  virtual void
  GPUGenerateData();
};

} // end namespace rtk

#  ifndef ITK_MANUAL_INSTANTIATION
#    include "rtkCudaBackProjectionImageFilter.hxx"
#  endif

#endif // end conditional definition of the class

#endif
