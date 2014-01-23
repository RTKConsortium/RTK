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
#include <itkImageFileWriter.h>

#include <rtkMacro.h>

rtk::CudaFFTRampImageFilter
 ::CudaFFTRampImageFilter()
 {
   // We use FFTW for the kernel so we need to do the same thing as in the parent
 #if defined(USE_FFTWF)
   this->SetGreatestPrimeFactor(13);
 #endif
 }

void
rtk::CudaFFTRampImageFilter::CudaPadInputImageRegion(const RegionType &inputRegion)
{
  UpdateTruncationMirrorWeights();
  RegionType paddedRegion = inputRegion;

  // Set x padding
  typename SizeType::SizeValueType xPaddedSize = 2*inputRegion.GetSize(0);
  while( GreatestPrimeFactor( xPaddedSize ) > m_GreatestPrimeFactor )
    xPaddedSize++;
  paddedRegion.SetSize(0, xPaddedSize);
  long zeroext = ( (long)xPaddedSize - (long)inputRegion.GetSize(0) ) / 2;
  paddedRegion.SetIndex(0, inputRegion.GetIndex(0) - zeroext);

  // Set y padding. Padding along Y is only required if
  // - there is some windowing in the Y direction
  // - the DFT requires the size to be the product of given prime factors
  typename SizeType::SizeValueType yPaddedSize = inputRegion.GetSize(1);
  if(this->GetHannCutFrequencyY()>0.)
    yPaddedSize *= 2;
  while( GreatestPrimeFactor( yPaddedSize ) > m_GreatestPrimeFactor )
    yPaddedSize++;
  paddedRegion.SetSize(1, yPaddedSize);
  paddedRegion.SetIndex(1, inputRegion.GetIndex(1) );

  // Create padded image (spacing and origin do not matter)
  itk::CudaImage<float,3>::Pointer paddedImage = itk::CudaImage<float,3>::New();
  paddedImage->SetRegions(paddedRegion);
  paddedImage->Allocate();
  paddedImage->FillBuffer(0);

  if(!m_TruncationMirrorWeights.size())
    m_TruncationMirrorWeights.push_back(0.);

  uint3 sz;
  long3 idx;
  idx.x = paddedRegion.GetIndex()[0];
  idx.y = paddedRegion.GetIndex()[1];
  idx.z = paddedRegion.GetIndex()[2];
  sz.x = paddedRegion.GetSize()[0];
  sz.y = paddedRegion.GetSize()[1];
  sz.z = paddedRegion.GetSize()[2];

  float *pin  = *(float**)( this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pout = *(float**)( paddedImage->GetCudaDataManager()->GetGPUBufferPointer() );

  // *** Call to Kernel *****************************************************************
  CUDA_padding(idx,
               sz,
               pin,
               pout,
               &m_TruncationMirrorWeights[0],
               m_TruncationMirrorWeights.size(),
               this->GetHannCutFrequencyY());

  m_padImage = paddedImage;
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
//  paddedImage = PadInputImageRegion<FFTInputImageType, FFTOutputImageType>(this->GetInput()->GetRequestedRegion());
  CudaPadInputImageRegion(this->GetInput()->GetRequestedRegion());
  int3 inputDimension;
  inputDimension.x = m_padImage->GetBufferedRegion().GetSize()[0];
  inputDimension.y = m_padImage->GetBufferedRegion().GetSize()[1];
  inputDimension.z = m_padImage->GetBufferedRegion().GetSize()[2];
  if(inputDimension.y==1 && inputDimension.z>1) // Troubles cuda 3.2 and 4.0
    std::swap(inputDimension.y, inputDimension.z);

  // Get FFT ramp kernel. Must be itk::Image because GetFFTRampKernel is not
  // compatible with itk::CudaImage + ITK 3.20.
  FFTOutputCPUImagePointer fftK;
  FFTOutputImageType::SizeType s = m_padImage->GetLargestPossibleRegion().GetSize();
  fftK = this->GetFFTRampKernel<FFTInputCPUImageType, FFTOutputCPUImageType>(s[0], s[1]);

  // Create the itk::CudaImage holding the kernel
  FFTOutputImageType::RegionType kreg = fftK->GetLargestPossibleRegion();
#if ITK_VERSION_MAJOR <= 3 && !defined(USE_FFTWF)
  kreg.SetSize(0, kreg.GetSize(0)/2+1);
#endif
  FFTOutputImagePointer fftKCUDA = FFTOutputImageType::New();
  fftKCUDA->SetRegions(kreg);
  fftKCUDA->Allocate();

  // CUFFT scales by the number of element, correct for it in kernel.
  // Also transfer the kernel from the itk::Image to the itk::CudaImage.
  itk::ImageRegionIterator<FFTOutputCPUImageType> itKI(fftK, kreg);
  itk::ImageRegionIterator<FFTOutputImageType>    itKO(fftKCUDA, kreg);
  FFTPrecisionType invNPixels = 1 / double(m_padImage->GetBufferedRegion().GetNumberOfPixels() );
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
                       *(float**)(m_padImage->GetCudaDataManager()->GetGPUBufferPointer()),
                       *(float2**)(fftKCUDA->GetCudaDataManager()->GetGPUBufferPointer()));

  // CUDA Cropping and Graft Output
  typedef rtk::CudaCropImageFilter CropFilter;
  CropFilter::Pointer cf = CropFilter::New();
  OutputImageType::SizeType upCropSize, lowCropSize;
  for(unsigned int i=0; i<OutputImageType::ImageDimension; i++)
    {
    lowCropSize[i] = this->GetOutput()->GetRequestedRegion().GetIndex()[i] -
                     m_padImage->GetLargestPossibleRegion().GetIndex()[i];
    upCropSize[i]  = m_padImage->GetLargestPossibleRegion().GetSize()[i] -
                     this->GetOutput()->GetRequestedRegion().GetSize()[i] -
                     lowCropSize[i];
    }
  cf->SetUpperBoundaryCropSize(upCropSize);
  cf->SetLowerBoundaryCropSize(lowCropSize);
  cf->SetInput(m_padImage);
  cf->Update();

  // We only want to graft the data. To do so, we copy the rest before grafting.
  cf->GetOutput()->CopyInformation(this->GetOutput());
  cf->GetOutput()->SetBufferedRegion(this->GetOutput()->GetBufferedRegion());
  cf->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
  this->GraftOutput(cf->GetOutput());
}
