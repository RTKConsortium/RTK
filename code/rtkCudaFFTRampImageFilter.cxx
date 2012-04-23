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
::GenerateData()
{
  this->AllocateOutputs();

  // Pad image region
  FFTInputImagePointer paddedImage = PadInputImageRegion(this->GetInput()->GetRequestedRegion() );

  int3 inputDimension;
  inputDimension.x = paddedImage->GetBufferedRegion().GetSize()[0];
  inputDimension.y = paddedImage->GetBufferedRegion().GetSize()[1];
  inputDimension.z = paddedImage->GetBufferedRegion().GetSize()[2];

  // Get FFT ramp kernel
  FFTOutputImagePointer fftK = this->GetFFTRampKernel(paddedImage->GetLargestPossibleRegion().GetSize(0) );

  // CUFFT scales by the number of element, correct for it in kernel
  itk::ImageRegionIterator<FFTOutputImageType> itK(fftK, fftK->GetBufferedRegion() );
  itK.GoToBegin();
  FFTPrecisionType invNPixels = 1 / double(paddedImage->GetBufferedRegion().GetNumberOfPixels() );
  while(!itK.IsAtEnd() ) {
    itK.Set(itK.Get() * invNPixels );
    ++itK;
    }

  CUDA_fft_convolution(inputDimension, paddedImage->GetBufferPointer(), (float2*)(fftK->GetBufferPointer() ) );

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
