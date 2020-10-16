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

#ifndef rtkCudaFDKBackProjectionImageFilter_h
#define rtkCudaFDKBackProjectionImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkFDKBackProjectionImageFilter.h"
#  include "RTKExport.h"

#  include <itkCudaImage.h>
#  include <itkCudaInPlaceImageFilter.h>

namespace rtk
{

/** \class CudaFDKBackProjectionImageFilter
 * \brief Cuda version of the FDK backprojection.
 *
 * GPU-based implementation of the backprojection step of the
 * [Feldkamp, Davis, Kress, 1984] algorithm for filtered backprojection
 * reconstruction of cone-beam CT images with a circular source trajectory.
 *
 * \author Simon Rit
 *
 * \ingroup RTK Projector CudaImageToImageFilter
 */
class RTK_EXPORT CudaFDKBackProjectionImageFilter
  : public itk::CudaInPlaceImageFilter<itk::CudaImage<float, 3>,
                                       itk::CudaImage<float, 3>,
                                       FDKBackProjectionImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>>
{
public:
#  if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(CudaFDKBackProjectionImageFilter);
#  else
  ITK_DISALLOW_COPY_AND_MOVE(CudaFDKBackProjectionImageFilter);
#  endif

  /** Standard class type alias. */
  using ImageType = itk::CudaImage<float, 3>;
  using FDKBackProjectionImageFilterType = FDKBackProjectionImageFilter<ImageType, ImageType>;
  using Self = CudaFDKBackProjectionImageFilter;
  using Superclass = itk::CudaInPlaceImageFilter<ImageType, ImageType, FDKBackProjectionImageFilterType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using OutputImageRegionType = ImageType::RegionType;
  using ProjectionImageType = itk::CudaImage<float, 2>;
  using ProjectionImagePointer = ProjectionImageType::Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaFDKBackProjectionImageFilter, Superclass);

protected:
  CudaFDKBackProjectionImageFilter();
  virtual ~CudaFDKBackProjectionImageFilter(){};

  virtual void
  GPUGenerateData();
};

} // end namespace rtk

#endif // end conditional definition of the class

#endif
