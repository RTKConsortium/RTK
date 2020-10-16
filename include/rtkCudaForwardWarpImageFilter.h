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

#ifndef rtkCudaForwardWarpImageFilter_h
#define rtkCudaForwardWarpImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "RTKExport.h"
#  include "rtkForwardWarpImageFilter.h"

#  include <itkCudaImage.h>
#  include <itkCudaImageToImageFilter.h>

namespace rtk
{

/** \class CudaForwardWarpImageFilter
 * \brief Cuda version of the ForwardWarpImageFilter
 *
 * Deform an image using a Displacement Vector Field, by performing
 * trilinear splat. Adjoint of the regular warp filter. GPU-based implementation
 *
 * \test rtkwarptest
 *
 * \author Cyril Mory
 *
 * \ingroup RTK CudaImageToImageFilter
 */
class RTK_EXPORT CudaForwardWarpImageFilter
  : public itk::CudaImageToImageFilter<itk::CudaImage<float, 3>,
                                       itk::CudaImage<float, 3>,
                                       rtk::ForwardWarpImageFilter<itk::CudaImage<float, 3>,
                                                                   itk::CudaImage<float, 3>,
                                                                   itk::CudaImage<itk::CovariantVector<float, 3>, 3>>>
{
public:
#  if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(CudaForwardWarpImageFilter);
#  else
  ITK_DISALLOW_COPY_AND_MOVE(CudaForwardWarpImageFilter);
#  endif

  /** Standard class type alias. */
  using ImageType = itk::CudaImage<float, 3>;
  using DisplacementVectorType = itk::CovariantVector<float, 3>;
  using DVFType = itk::CudaImage<DisplacementVectorType, 3>;
  using ForwardWarpImageFilterType = ForwardWarpImageFilter<ImageType, ImageType, DVFType>;
  using Self = CudaForwardWarpImageFilter;
  using Superclass = itk::CudaImageToImageFilter<ImageType, ImageType, ForwardWarpImageFilterType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using OutputImageRegionType = ImageType::RegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaForwardWarpImageFilter, Superclass);

protected:
  CudaForwardWarpImageFilter();
  virtual ~CudaForwardWarpImageFilter(){};

  virtual void
  GPUGenerateData();
};

} // end namespace rtk

#endif // end conditional definition of the class

#endif
