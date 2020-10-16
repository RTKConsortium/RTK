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

#ifndef rtkCudaWarpImageFilter_h
#define rtkCudaWarpImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "RTKExport.h"

#  include <itkCudaImage.h>
#  include <itkWarpImageFilter.h>
#  include <itkCudaImageToImageFilter.h>

namespace rtk
{

/** \class CudaWarpImageFilter
 * \brief Cuda version of the WarpImageFilter
 *
 * Deform an image using a Displacement Vector Field. GPU-based implementation
 *
 * \test rtkwarptest
 *
 * \author Cyril Mory
 *
 * \ingroup RTK CudaImageToImageFilter
 */
class RTK_EXPORT CudaWarpImageFilter
  : public itk::CudaImageToImageFilter<itk::CudaImage<float, 3>,
                                       itk::CudaImage<float, 3>,
                                       itk::WarpImageFilter<itk::CudaImage<float, 3>,
                                                            itk::CudaImage<float, 3>,
                                                            itk::CudaImage<itk::CovariantVector<float, 3>, 3>>>
{
public:
#  if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(CudaWarpImageFilter);
#  else
  ITK_DISALLOW_COPY_AND_MOVE(CudaWarpImageFilter);
#  endif

  /** Standard class type alias. */
  using ImageType = itk::CudaImage<float, 3>;
  using DisplacementVectorType = itk::CovariantVector<float, 3>;
  using DVFType = itk::CudaImage<DisplacementVectorType, 3>;
  using WarpImageFilterType = itk::WarpImageFilter<ImageType, ImageType, DVFType>;
  using Self = CudaWarpImageFilter;
  using Superclass = itk::CudaImageToImageFilter<ImageType, ImageType, WarpImageFilterType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using OutputImageRegionType = ImageType::RegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaWarpImageFilter, Superclass);

protected:
  CudaWarpImageFilter();
  virtual ~CudaWarpImageFilter(){};

  virtual void
  GPUGenerateData();
};

} // end namespace rtk

#endif // end conditional definition of the class

#endif
