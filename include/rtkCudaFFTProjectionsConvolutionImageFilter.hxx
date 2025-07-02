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

#ifndef rtkCudaFFTProjectionsConvolutionImageFilter_hxx
#define rtkCudaFFTProjectionsConvolutionImageFilter_hxx

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkCudaFFTProjectionsConvolutionImageFilter.h"
#  include "rtkCudaFFTProjectionsConvolutionImageFilter.hcu"

#  include "rtkCudaCropImageFilter.h"
#  include "rtkCudaCropImageFilter.hcu"

#  include <itkMacro.h>

namespace rtk
{

template <class TParentImageFilter>
CudaFFTProjectionsConvolutionImageFilter<TParentImageFilter>::CudaFFTProjectionsConvolutionImageFilter()
{
  // We use FFTW for the kernel so we need to do the same thing as in the parent
#  if defined(USE_FFTWF)
  this->SetGreatestPrimeFactor(13);
#  endif
}

template <class TParentImageFilter>
typename CudaFFTProjectionsConvolutionImageFilter<TParentImageFilter>::FFTInputImagePointer
CudaFFTProjectionsConvolutionImageFilter<TParentImageFilter>::PadInputImageRegion(const RegionType & inputRegion)
{
  CudaImageType::RegionType inBuffRegion = this->GetInput()->GetBufferedRegion();
  if (inBuffRegion != this->GetInput()->GetRequestedRegion())
  {
    itkExceptionMacro(
      << "CudaFFTProjectionsConvolutionImageFilter assumes that input requested and buffered regions are equal.");
  }

  TParentImageFilter::UpdateTruncationMirrorWeights();
  RegionType paddedRegion = TParentImageFilter::GetPaddedImageRegion(inputRegion);

  // Create padded image (spacing and origin do not matter)
  auto paddedImage = itk::CudaImage<float, 3>::New();
  paddedImage->SetRegions(paddedRegion);
  paddedImage->Allocate();

  uint3 sz, sz_i;
  int3  idx;
  idx.x = inBuffRegion.GetIndex()[0] - paddedRegion.GetIndex()[0];
  idx.y = inBuffRegion.GetIndex()[1] - paddedRegion.GetIndex()[1];
  idx.z = inBuffRegion.GetIndex()[2] - paddedRegion.GetIndex()[2];
  sz.x = paddedRegion.GetSize()[0];
  sz.y = paddedRegion.GetSize()[1];
  sz.z = paddedRegion.GetSize()[2];
  sz_i.x = inBuffRegion.GetSize()[0];
  sz_i.y = inBuffRegion.GetSize()[1];
  sz_i.z = inBuffRegion.GetSize()[2];

#  ifdef CudaCommon_VERSION_MAJOR
  float * pin = (float *)(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pout = (float *)(paddedImage->GetCudaDataManager()->GetGPUBufferPointer());
#  else
  float * pin = *(float **)(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pout = *(float **)(paddedImage->GetCudaDataManager()->GetGPUBufferPointer());
#  endif

  CUDA_padding(idx, sz, sz_i, pin, pout, TParentImageFilter::m_TruncationMirrorWeights);

  return paddedImage.GetPointer();
}

template <class TParentImageFilter>
void
CudaFFTProjectionsConvolutionImageFilter<TParentImageFilter>::GPUGenerateData()
{
  // Pad image region
  FFTInputImagePointer paddedImage = PadInputImageRegion(this->GetInput()->GetRequestedRegion());

  int3 inputDimension;
  inputDimension.x = paddedImage->GetBufferedRegion().GetSize()[0];
  inputDimension.y = paddedImage->GetBufferedRegion().GetSize()[1];
  inputDimension.z = paddedImage->GetBufferedRegion().GetSize()[2];
  if (inputDimension.y == 1 && inputDimension.z > 1) // Troubles cuda 3.2 and 4.0
    std::swap(inputDimension.y, inputDimension.z);

  // Get FFT ramp kernel. Must be itk::Image because GetFFTProjectionsConvolutionKernel is not
  // compatible with itk::CudaImage + ITK 3.20.
  typename Superclass::FFTOutputImageType::SizeType s = paddedImage->GetLargestPossibleRegion().GetSize();
  this->UpdateFFTProjectionsConvolutionKernel(s);
  if (this->m_KernelFFTCUDA.GetPointer() == nullptr ||
      this->m_KernelFFTCUDA->GetTimeStamp() < this->m_KernelFFT->GetTimeStamp())
  {

    // Create the itk::CudaImage holding the kernel
    typename Superclass::FFTOutputImageType::RegionType kreg = this->m_KernelFFT->GetLargestPossibleRegion();

    this->m_KernelFFTCUDA = CudaFFTOutputImageType::New();
    this->m_KernelFFTCUDA->SetRegions(kreg);
    this->m_KernelFFTCUDA->Allocate();

    // CUFFT scales by the number of element, correct for it in kernel.
    // Also transfer the kernel from the itk::Image to the itk::CudaImage.
    itk::ImageRegionIterator<typename TParentImageFilter::FFTOutputImageType> itKI(this->m_KernelFFT, kreg);
    itk::ImageRegionIterator<CudaFFTOutputImageType>                          itKO(this->m_KernelFFTCUDA, kreg);
    typename TParentImageFilter::FFTPrecisionType                             invNPixels;
    invNPixels = 1 / double(paddedImage->GetBufferedRegion().GetNumberOfPixels());
    while (!itKO.IsAtEnd())
    {
      itKO.Set(itKI.Get() * invNPixels);
      ++itKI;
      ++itKO;
    }
  }

  CudaImageType * cuPadImgP = dynamic_cast<CudaImageType *>(paddedImage.GetPointer());

  int2 kernelDimension;
  kernelDimension.x = this->m_KernelFFT->GetBufferedRegion().GetSize()[0];
  kernelDimension.y = this->m_KernelFFT->GetBufferedRegion().GetSize()[1];
  CUDA_fft_convolution(inputDimension,
                       kernelDimension,
#  ifdef CudaCommon_VERSION_MAJOR
                       (float *)(cuPadImgP->GetCudaDataManager()->GetGPUBufferPointer()),
                       (float2 *)(this->m_KernelFFTCUDA->GetCudaDataManager()->GetGPUBufferPointer()));
#  else
                       *(float **)(cuPadImgP->GetCudaDataManager()->GetGPUBufferPointer()),
                       *(float2 **)(this->m_KernelFFTCUDA->GetCudaDataManager()->GetGPUBufferPointer()));
#  endif

  /* Impossible to get a pixel-wise progress reporting with CUDA
   * so this filter only reports 0 and 100% completion. */
  itk::ProgressReporter progress(this, 0, 1);
  progress.CompletedPixel();

  // CUDA Cropping and Graft Output
  auto                                           cf = CudaCropImageFilter::New();
  typename Superclass::OutputImageType::SizeType upCropSize, lowCropSize;
  for (unsigned int i = 0; i < CudaImageType::ImageDimension; i++)
  {
    lowCropSize[i] =
      this->GetOutput()->GetRequestedRegion().GetIndex()[i] - paddedImage->GetLargestPossibleRegion().GetIndex()[i];
    upCropSize[i] = paddedImage->GetLargestPossibleRegion().GetSize()[i] -
                    this->GetOutput()->GetRequestedRegion().GetSize()[i] - lowCropSize[i];
  }
  cf->SetUpperBoundaryCropSize(upCropSize);
  cf->SetLowerBoundaryCropSize(lowCropSize);
  cf->SetInput(cuPadImgP);
  cf->Update();

  // We only want to graft the data. To do so, we copy the rest before grafting.
  cf->GetOutput()->CopyInformation(this->GetOutput());
  cf->GetOutput()->SetBufferedRegion(this->GetOutput()->GetBufferedRegion());
  cf->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
  this->GraftOutput(cf->GetOutput());
}

} // namespace rtk

#endif // end conditional definition of the class

#endif
