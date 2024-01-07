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

#ifndef rtkCudaSplatImageFilter_h
#define rtkCudaSplatImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkSplatWithKnownWeightsImageFilter.h"
#  include "itkCudaImage.h"
#  include "itkCudaInPlaceImageFilter.h"
#  include "RTKExport.h"

namespace rtk
{

/** \class CudaSplatImageFilter
 * \brief Implements the SplatWithKnownWeightsImageFilter on GPU.
 *
 *
 *
 * \author Cyril Mory
 *
 * \ingroup RTK CudaImageToImageFilter
 */
class RTK_EXPORT CudaSplatImageFilter
  : public itk::CudaInPlaceImageFilter<
      itk::CudaImage<float, 4>,
      itk::CudaImage<float, 4>,
      SplatWithKnownWeightsImageFilter<itk::CudaImage<float, 4>, itk::CudaImage<float, 3>>>

{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaSplatImageFilter);

  /** Standard class type alias. */
  using Self = rtk::CudaSplatImageFilter;
  using Superclass = rtk::SplatWithKnownWeightsImageFilter<OutputImageType, InputImageType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
#  ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(CudaSplatImageFilter);
#  else
  itkTypeMacro(CudaSplatImageFilter, SplatWithKnownWeightsImageFilter);
#  endif

protected:
  CudaSplatImageFilter();
  ~CudaSplatImageFilter() {}

  virtual void
  GPUGenerateData();

}; // end of class

} // end namespace rtk

#endif // end conditional definition of the class

#endif
