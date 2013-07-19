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

#include "rtkCudaFFTRampImageFilter.h"
#include "rtkCudaFFTRampImageFilter.hcu"

#include <itkMacro.h>

rtk::CudaFFTRampImageFilter
 ::CudaFFTRampImageFilter()
 {
   // We use FFTW for the kernel so we need to do the same thing as in the parent
 #if defined(USE_FFTWF)
   this->SetGreatestPrimeFactor(13);
 #endif
 }

void
rtk::CudaFFTRampImageFilter
::GPUGenerateData()
{
  typedef itk::CudaImage<float,
                         ImageType::ImageDimension > FFTInputImageType;
  typedef FFTInputImageType::Pointer                 FFTInputImagePointer;
  typedef itk::CudaImage<std::complex<float>,
                         ImageType::ImageDimension > FFTOutputImageType;
  typedef FFTOutputImageType::Pointer                FFTOutputImagePointer;

  this->AllocateOutputs();

  // Pad image region
  FFTInputImagePointer paddedImage;
  paddedImage = PadInputImageRegion<FFTInputImageType, FFTOutputImageType>(this->GetInput()->GetRequestedRegion());

  int3 inputDimension;
  inputDimension.x = paddedImage->GetBufferedRegion().GetSize()[0];
  inputDimension.y = paddedImage->GetBufferedRegion().GetSize()[1];
  inputDimension.z = paddedImage->GetBufferedRegion().GetSize()[2];
  if(inputDimension.y==1 && inputDimension.z>1) // Troubles cuda 3.2 and 4.0
    std::swap(inputDimension.y, inputDimension.z);

  // Get FFT ramp kernel
  FFTOutputImagePointer fftK;
  FFTOutputImageType::SizeType s = paddedImage->GetLargestPossibleRegion().GetSize();
  fftK = this->GetFFTRampKernel<FFTInputImageType, FFTOutputImageType>(s[0], s[1]);

  // CUFFT scales by the number of element, correct for it in kernel
  itk::ImageRegionIterator<FFTOutputImageType> itK(fftK, fftK->GetBufferedRegion() );
  itK.GoToBegin();
  FFTPrecisionType invNPixels = 1 / double(paddedImage->GetBufferedRegion().GetNumberOfPixels() );
  while(!itK.IsAtEnd() ) {
    itK.Set(itK.Get() * invNPixels );
    ++itK;
    }

  int2 kernelDimension;
  kernelDimension.x = fftK->GetBufferedRegion().GetSize()[0];
  kernelDimension.y = fftK->GetBufferedRegion().GetSize()[1];
  CUDA_fft_convolution(inputDimension,
                       kernelDimension,
                       *(float**)(paddedImage->GetCudaDataManager()->GetGPUBufferPointer()),
                       *(float2**)(fftK->GetCudaDataManager()->GetGPUBufferPointer()));

  // Crop and paste result
  itk::ImageRegionConstIterator<FFTInputImageType> itS(paddedImage, this->GetOutput()->GetRequestedRegion() );
  itk::ImageRegionIterator<OutputImageType>        itD(this->GetOutput(), this->GetOutput()->GetRequestedRegion() );
  itS.GoToBegin();
  itD.GoToBegin();
  while(!itS.IsAtEnd() ) {
    itD.Set(itS.Get() );
    ++itS;
    ++itD;
    }
}
