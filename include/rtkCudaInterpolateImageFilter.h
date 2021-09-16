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

#ifndef rtkCudaInterpolateImageFilter_h
#define rtkCudaInterpolateImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkInterpolatorWithKnownWeightsImageFilter.h"
#  include "itkCudaImage.h"
#  include "itkCudaInPlaceImageFilter.h"
#  include "RTKExport.h"

namespace rtk
{

/** \class CudaInterpolateImageFilter
 * \brief Implements the InterpolateWithKnownWeightsImageFilter on GPU.
 *
 *
 *
 * \author Cyril Mory
 *
 * \ingroup RTK CudaImageToImageFilter
 */
class RTK_EXPORT CudaInterpolateImageFilter
  : public itk::CudaInPlaceImageFilter<
      itk::CudaImage<float, 3>,
      itk::CudaImage<float, 3>,
      InterpolatorWithKnownWeightsImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 4>>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaInterpolateImageFilter);

  /** Standard class type alias. */
  using Self = rtk::CudaInterpolateImageFilter;
  using Superclass = rtk::InterpolatorWithKnownWeightsImageFilter<OutputImageType, InputImageType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaInterpolateImageFilter, InterpolatorWithKnownWeightsImageFilter);

protected:
  CudaInterpolateImageFilter();
  ~CudaInterpolateImageFilter() {}

  virtual void
  GPUGenerateData();

}; // end of class

} // end namespace rtk

#endif // end conditional definition of the class

#endif
