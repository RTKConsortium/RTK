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

#include "rtkCudaCropImageFilter.h"
#include "rtkCudaCropImageFilter.hcu"

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
  // Cuda typedefs
  typedef itk::CudaImage<float,
                         ImageType::ImageDimension > FFTInputImageType;
  typedef FFTInputImageType::Pointer                 FFTInputImagePointer;
  typedef itk::CudaImage<std::complex<float>,
                         ImageType::ImageDimension > FFTOutputImageType;
  typedef FFTOutputImageType::Pointer                FFTOutputImagePointer;

  // Non-cuda typedefs
  typedef itk::Image<float,
                     ImageType::ImageDimension >     FFTInputCPUImageType;
  typedef FFTInputCPUImageType::Pointer              FFTInputCPUImagePointer;
  typedef itk::Image<std::complex<float>,
                     ImageType::ImageDimension >     FFTOutputCPUImageType;
  typedef FFTOutputCPUImageType::Pointer             FFTOutputCPUImagePointer;

  //this->AllocateOutputs();

  // Pad image region
  FFTInputImagePointer paddedImage;
  paddedImage = PadInputImageRegion<FFTInputImageType, FFTOutputImageType>(this->GetInput()->GetRequestedRegion());

  int3 inputDimension;
  inputDimension.x = paddedImage->GetBufferedRegion().GetSize()[0];
  inputDimension.y = paddedImage->GetBufferedRegion().GetSize()[1];
  inputDimension.z = paddedImage->GetBufferedRegion().GetSize()[2];
  if(inputDimension.y==1 && inputDimension.z>1) // Troubles cuda 3.2 and 4.0
    std::swap(inputDimension.y, inputDimension.z);

  // Get FFT ramp kernel. Must be itk::Image because GetFFTRampKernel is not
  // compatible with itk::CudaImage + ITK 3.20.
  FFTOutputCPUImagePointer fftK;
  FFTOutputImageType::SizeType s = paddedImage->GetLargestPossibleRegion().GetSize();
  fftK = this->GetFFTRampKernel<FFTInputCPUImageType, FFTOutputCPUImageType>(s[0], s[1]);

  // Create the itk::CudaImage holding the kernel
  FFTOutputImageType::RegionType kreg = fftK->GetLargestPossibleRegion();
  FFTOutputImagePointer fftKCUDA = FFTOutputImageType::New();
  fftKCUDA->SetRegions(kreg);
  fftKCUDA->Allocate();

  // CUFFT scales by the number of element, correct for it in kernel.
  // Also transfer the kernel from the itk::Image to the itk::CudaImage.
  itk::ImageRegionIterator<FFTOutputCPUImageType> itKI(fftK, kreg);
  itk::ImageRegionIterator<FFTOutputImageType>    itKO(fftKCUDA, kreg);
  FFTPrecisionType invNPixels = 1 / double(paddedImage->GetBufferedRegion().GetNumberOfPixels() );
  while(!itKO.IsAtEnd() )
    {
    itKO.Set(itKI.Get() * invNPixels );
    ++itKI;
    ++itKO;
    }

  int2 kernelDimension;
  kernelDimension.x = fftK->GetBufferedRegion().GetSize()[0];
  kernelDimension.y = fftK->GetBufferedRegion().GetSize()[1];
  CUDA_fft_convolution(inputDimension,
                       kernelDimension,
                       *(float**)(paddedImage->GetCudaDataManager()->GetGPUBufferPointer()),
                       *(float2**)(fftKCUDA->GetCudaDataManager()->GetGPUBufferPointer()));

  // CUDA Cropping and Graft Output
  typedef rtk::CudaCropImageFilter CropFilter;
  CropFilter::Pointer cf = CropFilter::New();
  OutputImageType::SizeType upCropSize, lowCropSize;
  for(unsigned int i=0; i<OutputImageType::ImageDimension; i++)
    {
    lowCropSize[i] = this->GetOutput()->GetRequestedRegion().GetIndex()[i] -
                     paddedImage->GetLargestPossibleRegion().GetIndex()[i];
    upCropSize[i]  = paddedImage->GetLargestPossibleRegion().GetSize()[i] -
                     this->GetOutput()->GetRequestedRegion().GetSize()[i] -
                     lowCropSize[i];
    }
  cf->SetUpperBoundaryCropSize(upCropSize);
  cf->SetLowerBoundaryCropSize(lowCropSize);
  cf->SetInput(paddedImage);
  cf->Update();

  // We only want to graft the data. To do so, we copy the rest before grafting.
  cf->GetOutput()->CopyInformation(this->GetOutput());
  cf->GetOutput()->SetBufferedRegion(this->GetOutput()->GetBufferedRegion());
  cf->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
  this->GraftOutput(cf->GetOutput());
}
