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

#ifndef rtkCudaFFTProjectionsConvolutionImageFilter_h
#define rtkCudaFFTProjectionsConvolutionImageFilter_h

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include <itkCudaImage.h>
#  include <itkCudaImageToImageFilter.h>

namespace rtk
{

/** \class CudaFFTProjectionsConvolutionImageFilter
 * \brief Implements 1D or 2D FFT convolution.
 *
 * This filter implements a convolution using FFT of the input image. The
 * convolution kernel must be defined in the parent class, passed via the
 * template argument. The template argument must be a child of
 * rtk::FFTProjectionsConvolutionImageFilter.
 *
 * \see rtk::FFTProjectionsConvolutionImageFilter
 *
 * \test rtkrampfiltertest.cxx, rtkrampfiltertest2.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK CudaImageToImageFilter
 */
template <class TParentImageFilter>
class CudaFFTProjectionsConvolutionImageFilter
  : public itk::CudaImageToImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>, TParentImageFilter>
{
public:
#  if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(CudaFFTProjectionsConvolutionImageFilter);
#  else
  ITK_DISALLOW_COPY_AND_MOVE(CudaFFTProjectionsConvolutionImageFilter);
#  endif

  /** Standard class type alias. */
  using Self = CudaFFTProjectionsConvolutionImageFilter;
  using Superclass = TParentImageFilter;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  using RegionType = typename TParentImageFilter::RegionType;
  using FFTInputImagePointer = typename TParentImageFilter::FFTInputImagePointer;
  using CudaImageType = itk::CudaImage<float, 3>;
  using CudaFFTOutputImageType = itk::CudaImage<std::complex<float>, 3>;
  using CudaFFTOutputImagePointer = CudaFFTOutputImageType::Pointer;

  /** Runtime information support. */
  itkTypeMacro(CudaFFTProjectionsConvolutionImageFilter, TParentImageFilter);

protected:
  CudaFFTProjectionsConvolutionImageFilter();
  ~CudaFFTProjectionsConvolutionImageFilter() {}

  virtual void
  GPUGenerateData();

  virtual FFTInputImagePointer
  PadInputImageRegion(const RegionType & inputRegion);

private:
  CudaFFTOutputImagePointer m_KernelFFTCUDA;
}; // end of class

} // end namespace rtk

#  ifndef ITK_MANUAL_INSTANTIATION
#    include "rtkCudaFFTProjectionsConvolutionImageFilter.hxx"
#  endif

#endif // end conditional definition of the class

#endif
