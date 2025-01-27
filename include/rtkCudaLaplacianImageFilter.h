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

#ifndef rtkCudaLaplacianImageFilter_h
#define rtkCudaLaplacianImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkLaplacianImageFilter.h"
#  include <itkCudaInPlaceImageFilter.h>
#  include "RTKExport.h"

namespace rtk
{

/** \class CudaLaplacianImageFilter
 * \brief Implements the 3D float LaplacianImageFilter on GPU
 *
 * \author Cyril Mory
 *
 * \ingroup RTK CudaImageToImageFilter
 */

class RTK_EXPORT CudaLaplacianImageFilter
  : public itk::CudaImageToImageFilter<
      itk::CudaImage<float, 3>,
      itk::CudaImage<float, 3>,
      LaplacianImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<itk::CovariantVector<float, 3>, 3>>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaLaplacianImageFilter);

  /** Standard class type alias. */
  using Self = rtk::CudaLaplacianImageFilter;
  using OutputImageType = itk::CudaImage<float, 3>;
  using GradientImageType = itk::CudaImage<itk::CovariantVector<float, 3>, 3>;
  using Superclass = rtk::LaplacianImageFilter<OutputImageType, GradientImageType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(CudaLaplacianImageFilter);

protected:
  CudaLaplacianImageFilter();
  ~CudaLaplacianImageFilter() {}

  virtual void
  GPUGenerateData();

}; // end of class

} // end namespace rtk

#endif // end conditional definition of the class

#endif
