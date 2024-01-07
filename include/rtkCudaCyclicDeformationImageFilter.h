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

#ifndef rtkCudaCyclicDeformationImageFilter_h
#define rtkCudaCyclicDeformationImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkCyclicDeformationImageFilter.h"
#  include <itkCudaImageToImageFilter.h>
#  include "RTKExport.h"

namespace rtk
{

/** \class CudaCyclicDeformationImageFilter
 * \brief GPU version of the temporal DVF interpolator
 *
 * This filter implements linear interpolation along time
 * in a DVF, assuming that the motion is periodic
 *
 * \author Cyril Mory
 *
 * \ingroup RTK CudaImageToImageFilter
 */

class RTK_EXPORT CudaCyclicDeformationImageFilter
  : public itk::CudaImageToImageFilter<itk::CudaImage<itk::CovariantVector<float, 3>, 4>,
                                       itk::CudaImage<itk::CovariantVector<float, 3>, 3>,
                                       CyclicDeformationImageFilter<itk::CudaImage<itk::CovariantVector<float, 3>, 4>,
                                                                    itk::CudaImage<itk::CovariantVector<float, 3>, 3>>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaCyclicDeformationImageFilter);

  /** Standard class type alias. */
  using Self = rtk::CudaCyclicDeformationImageFilter;
  using InputImageType = itk::CudaImage<itk::CovariantVector<float, 3>, 4>;
  using OutputImageType = itk::CudaImage<itk::CovariantVector<float, 3>, 3>;
  using Superclass = rtk::CyclicDeformationImageFilter<InputImageType, OutputImageType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
#  ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(CudaCyclicDeformationImageFilter);
#  else
  itkTypeMacro(CudaCyclicDeformationImageFilter, CyclicDeformationImageFilter);
#  endif

protected:
  CudaCyclicDeformationImageFilter();
  ~CudaCyclicDeformationImageFilter() {}

  virtual void
  GPUGenerateData();

}; // end of class

} // end namespace rtk

#endif // end conditional definition of the class

#endif
