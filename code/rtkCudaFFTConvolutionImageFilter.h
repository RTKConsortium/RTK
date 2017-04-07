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

#ifndef rtkCudaFFTConvolutionImageFilter_h
#define rtkCudaFFTConvolutionImageFilter_h

#include <itkCudaImage.h>
#include <itkCudaImageToImageFilter.h>

namespace rtk
{

/** \class CudaFFTConvolutionImageFilter
 * \brief Implements 1D or 2D FFT convolution.
 *
 * This filter implements a convolution using FFT of the input image. The
 * convolution kernel must be defined in the parent class, passed via the
 * template argument. The template argument must be a child of
 * rtk::FFTConvolutionImageFilter.
 *
 * \see rtk::FFTConvolutionImageFilter
 *
 * \test rtkrampfiltertest.cxx, rtkrampfiltertest2.cxx
 *
 * \author Simon Rit
 *
 * \ingroup CudaImageToImageFilter
 */
template< class TParentImageFilter >
class ITK_EXPORT CudaFFTConvolutionImageFilter:
  public itk::CudaImageToImageFilter< itk::CudaImage<float,3>,
                                      itk::CudaImage<float,3>,
                                      TParentImageFilter >
{
public:
  /** Standard class typedefs. */
  typedef CudaFFTConvolutionImageFilter Self;
  typedef TParentImageFilter            Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Convenient typedefs. */
  typedef typename TParentImageFilter::RegionType           RegionType;
  typedef typename TParentImageFilter::FFTInputImagePointer FFTInputImagePointer;
  typedef itk::CudaImage<float,3>                           CudaImageType;
  typedef itk::CudaImage< std::complex<float>, 3 >          CudaFFTOutputImageType;
  typedef CudaFFTOutputImageType::Pointer                   CudaFFTOutputImagePointer;

  /** Runtime information support. */
  itkTypeMacro(CudaFFTConvolutionImageFilter, TParentImageFilter);

protected:
  CudaFFTConvolutionImageFilter();
  ~CudaFFTConvolutionImageFilter(){}

  virtual void GPUGenerateData();

  virtual FFTInputImagePointer PadInputImageRegion(const RegionType &inputRegion);

private:
  CudaFFTConvolutionImageFilter(const Self&); // purposely not implemented
  void operator=(const Self&);                // purposely not implemented

  CudaFFTOutputImagePointer m_KernelFFTCUDA;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkCudaFFTConvolutionImageFilter.hxx"
#endif

#endif
