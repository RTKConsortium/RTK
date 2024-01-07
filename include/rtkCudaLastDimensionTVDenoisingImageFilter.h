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

#ifndef rtkCudaLastDimensionTVDenoisingImageFilter_h
#define rtkCudaLastDimensionTVDenoisingImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkTotalVariationDenoisingBPDQImageFilter.h"
#  include <itkCudaInPlaceImageFilter.h>
#  include "RTKExport.h"

namespace rtk
{

/** \class CudaLastDimensionTVDenoisingImageFilter
 * \brief Implements the TotalVariationDenoisingBPDQImageFilter on GPU
 * for a specific case : denoising only along the last dimension
 *
 *
 *
 * \author Cyril Mory
 *
 * \ingroup RTK CudaImageToImageFilter
 */

class RTK_EXPORT CudaLastDimensionTVDenoisingImageFilter
  : public itk::CudaInPlaceImageFilter<
      itk::CudaImage<float, 4>,
      itk::CudaImage<float, 4>,
      TotalVariationDenoisingBPDQImageFilter<itk::CudaImage<float, 4>,
                                             itk::CudaImage<itk::CovariantVector<float, 1>, 4>>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaLastDimensionTVDenoisingImageFilter);

  /** Standard class type alias. */
  using Self = rtk::CudaLastDimensionTVDenoisingImageFilter;
  using OutputImageType = itk::CudaImage<float, 4>;
  using GradientType = itk::CudaImage<itk::CovariantVector<float, 1>, 4>;
  using Superclass = rtk::TotalVariationDenoisingBPDQImageFilter<OutputImageType, GradientType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
#  ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(CudaLastDimensionTVDenoisingImageFilter);
#  else
  itkTypeMacro(CudaLastDimensionTVDenoisingImageFilter, TotalVariationDenoisingBPDQImageFilter);
#  endif

protected:
  CudaLastDimensionTVDenoisingImageFilter();
  ~CudaLastDimensionTVDenoisingImageFilter() {}

  virtual void
  GPUGenerateData();

}; // end of class

} // end namespace rtk

#endif // end conditional definition of the class

#endif
