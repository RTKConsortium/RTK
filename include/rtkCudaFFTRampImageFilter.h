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

#ifndef rtkCudaFFTRampImageFilter_h
#define rtkCudaFFTRampImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkCudaFFTProjectionsConvolutionImageFilter.h"
#  include "rtkFFTRampImageFilter.h"
#  include "RTKExport.h"

namespace rtk
{

/** \class CudaFFTRampImageFilter
 * \brief Implements the ramp image filter of the FDK algorithm on GPU.
 *
 * Uses CUFFT for the projection fft and ifft.
 *
 * \author Simon Rit
 *
 * \ingroup RTK CudaImageToImageFilter
 */
class RTK_EXPORT CudaFFTRampImageFilter
  : public CudaFFTProjectionsConvolutionImageFilter<
      FFTRampImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>, float>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaFFTRampImageFilter);

  /** Standard class type alias. */
  using Self = CudaFFTRampImageFilter;
  using Superclass = FFTRampImageFilter<CudaImageType, CudaImageType, float>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(CudaFFTRampImageFilter);

protected:
  CudaFFTRampImageFilter() {}
  ~CudaFFTRampImageFilter() override = default;

}; // end of class

} // end namespace rtk

#endif // end conditional definition of the class

#endif
