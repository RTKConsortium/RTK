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

#ifndef rtkFFTProjectionsConvolutionImageFilter_hxx
#define rtkFFTProjectionsConvolutionImageFilter_hxx


// Use local RTK FFTW files taken from Gaëtan Lehmann's code for
// thread safety: https://hdl.handle.net/10380/3154
#include <itkRealToHalfHermitianForwardFFTImageFilter.h>
#include <itkHalfHermitianToRealInverseFFTImageFilter.h>

#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

namespace rtk
{

template <class TInputImage, class TOutputImage, class TFFTPrecision>
FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::FFTProjectionsConvolutionImageFilter()
{
  this->DynamicMultiThreadingOff();
#if defined(USE_FFTWD)
  if (typeid(TFFTPrecision).name() == typeid(double).name())
    m_GreatestPrimeFactor = 13;
#endif
#if defined(USE_FFTWF)
  if (typeid(TFFTPrecision).name() == typeid(float).name())
    m_GreatestPrimeFactor = 13;
#endif

  m_ZeroPadFactors.Fill(2);
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  auto * input = const_cast<InputImageType *>(this->GetInput());
  if (!input)
    return;

  // Compute input region (==requested region fully enlarged for dim 0)
  RegionType inputRegion;
  this->CallCopyOutputRegionToInputRegion(inputRegion, this->GetOutput()->GetRequestedRegion());
  inputRegion.SetIndex(0, this->GetOutput()->GetLargestPossibleRegion().GetIndex(0));
  inputRegion.SetSize(0, this->GetOutput()->GetLargestPossibleRegion().GetSize(0));

  // Also enlarge along dim 1 if 2D kernel is used
  if (m_KernelDimension == 2)
  {
    inputRegion.SetIndex(1, this->GetOutput()->GetLargestPossibleRegion().GetIndex(1));
    inputRegion.SetSize(1, this->GetOutput()->GetLargestPossibleRegion().GetSize(1));
  }
  input->SetRequestedRegion(inputRegion);
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
int
FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::GetTruncationCorrectionExtent()
{
  return itk::Math::floor(m_TruncationCorrection * this->GetInput()->GetRequestedRegion().GetSize(0));
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::BeforeThreadedGenerateData()
{
  UpdateTruncationMirrorWeights();

  // If the following condition is met, multi-threading is left to the (i)fft
  // filter. Otherwise, one splits the image and a separate fft is performed
  // per thread.
  auto         reqRegion = this->GetOutput()->GetRequestedRegion();
  unsigned int nproj = reqRegion.GetNumberOfPixels() / (reqRegion.GetSize()[0] * reqRegion.GetSize()[1]);
  if ((nproj == 1 && m_KernelDimension == 2) || (reqRegion.GetSize()[0] == reqRegion.GetNumberOfPixels()))
  {
    m_BackupNumberOfThreads = this->GetNumberOfWorkUnits();
    this->SetNumberOfWorkUnits(1);
  }
  else
    m_BackupNumberOfThreads = 1;

  // Update FFT ramp kernel (if required)
  RegionType paddedRegion = GetPaddedImageRegion(this->GetInput()->GetRequestedRegion());
  UpdateFFTProjectionsConvolutionKernel(paddedRegion.GetSize());
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::AfterThreadedGenerateData()
{
  auto         reqRegion = this->GetOutput()->GetRequestedRegion();
  unsigned int nproj = reqRegion.GetNumberOfPixels() / (reqRegion.GetSize()[0] * reqRegion.GetSize()[1]);
  if ((nproj == 1 && m_KernelDimension == 2) || (reqRegion.GetSize()[0] == reqRegion.GetNumberOfPixels()))
  {
    this->SetNumberOfWorkUnits(m_BackupNumberOfThreads);
  }
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::ThreadedGenerateData(
  const RegionType & outputRegionForThread,
  ThreadIdType       threadId)
{
  auto nproj = outputRegionForThread.GetNumberOfPixels() /
               (outputRegionForThread.GetSize()[0] * outputRegionForThread.GetSize()[1]);

  itk::ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels(), 100);

  for (unsigned int i = 0; i < nproj; i++)
  {
    auto outputRegion = outputRegionForThread;
    if (outputRegion.GetImageDimension() == 3)
    {
      outputRegion.SetSize(2, 1);
      outputRegion.SetIndex(2, outputRegionForThread.GetIndex(2) + i);
    }
    else if (outputRegion.GetImageDimension() == 4)
    {
      outputRegion.SetSize(2, 1);
      outputRegion.SetSize(3, 1);
      outputRegion.SetIndex(2, outputRegionForThread.GetIndex(2) + i % outputRegionForThread.GetSize(2));
      outputRegion.SetIndex(3, outputRegionForThread.GetIndex(3) + i / outputRegionForThread.GetSize(2));
    }

    // Pad image region enlarged along X
    RegionType enlargedRegionX = outputRegion;
    enlargedRegionX.SetIndex(0, this->GetInput()->GetRequestedRegion().GetIndex(0));
    enlargedRegionX.SetSize(0, this->GetInput()->GetRequestedRegion().GetSize(0));
    enlargedRegionX.SetIndex(1, this->GetInput()->GetRequestedRegion().GetIndex(1));
    enlargedRegionX.SetSize(1, this->GetInput()->GetRequestedRegion().GetSize(1));
    FFTInputImagePointer paddedImage;
    paddedImage = PadInputImageRegion(enlargedRegionX);

    // FFT padded image
    using FFTType = itk::RealToHalfHermitianForwardFFTImageFilter<FFTInputImageType>;
    auto fftI = FFTType::New();
    fftI->SetInput(paddedImage);
    fftI->SetNumberOfWorkUnits(m_BackupNumberOfThreads);
    fftI->Update();

    // Multiply line-by-line or projection-by-projection (depends on kernel size)
    itk::ImageRegionIterator<typename FFTType::OutputImageType> itI(fftI->GetOutput(),
                                                                    fftI->GetOutput()->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<FFTOutputImageType> itK(m_KernelFFT, m_KernelFFT->GetLargestPossibleRegion());
    itI.GoToBegin();
    while (!itI.IsAtEnd())
    {
      itK.GoToBegin();
      while (!itK.IsAtEnd())
      {
        itI.Set(itI.Get() * itK.Get());
        ++itI;
        ++itK;
      }
    }

    // Inverse FFT image
    auto ifft = itk::HalfHermitianToRealInverseFFTImageFilter<typename FFTType::OutputImageType>::New();
    ifft->SetInput(fftI->GetOutput());
    ifft->SetNumberOfWorkUnits(m_BackupNumberOfThreads);
    ifft->SetReleaseDataFlag(true);
    ifft->SetActualXDimensionIsOdd(paddedImage->GetLargestPossibleRegion().GetSize(0) % 2);
    ifft->Update();

    // Crop and paste result
    itk::ImageRegionConstIterator<FFTInputImageType> itS(ifft->GetOutput(), outputRegion);
    itk::ImageRegionIterator<OutputImageType>        itD(this->GetOutput(), outputRegion);
    itS.GoToBegin();
    itD.GoToBegin();
    while (!itS.IsAtEnd())
    {
      itD.Set(itS.Get());
      ++itS;
      ++itD;
      progress.CompletedPixel();
    }
  }
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
typename FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::FFTInputImagePointer
FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::PadInputImageRegion(
  const RegionType & inputRegion)
{
  UpdateTruncationMirrorWeights();
  RegionType paddedRegion = GetPaddedImageRegion(inputRegion);

  // Create padded image (spacing and origin do not matter)
  FFTInputImagePointer paddedImage = FFTInputImageType::New();
  paddedImage->SetRegions(paddedRegion);
  paddedImage->Allocate();
  paddedImage->FillBuffer(0);

  const long next = std::min(inputRegion.GetIndex(0) - paddedRegion.GetIndex(0),
                             (typename FFTInputImageType::IndexValueType)this->GetTruncationCorrectionExtent());
  if (next)
  {
    typename FFTInputImageType::IndexType                 idx;
    typename FFTInputImageType::IndexType                 iidx; // Reference point (SA / SE)
    typename FFTInputImageType::IndexType::IndexValueType borderDist = 0, rightIdx = 0;

    // Mirror left (equation 3a in [Ohnesorge et al, Med Phys, 2000])
    RegionType leftRegion = inputRegion;
    leftRegion.SetIndex(0, inputRegion.GetIndex(0) - next);
    leftRegion.SetSize(0, next);
    itk::ImageRegionIteratorWithIndex<FFTInputImageType> itLeft(paddedImage, leftRegion);
    while (!itLeft.IsAtEnd())
    {
      iidx = itLeft.GetIndex();
      iidx[0] = leftRegion.GetIndex(0) + leftRegion.GetSize(0) + 1;
      TFFTPrecision SA = this->GetInput()->GetPixel(iidx);
      for (unsigned int i = 0; i < leftRegion.GetSize(0); i++, ++itLeft)
      {
        idx = itLeft.GetIndex();
        borderDist = inputRegion.GetIndex(0) - idx[0];
        idx[0] = inputRegion.GetIndex(0) + borderDist;
        itLeft.Set(m_TruncationMirrorWeights[borderDist] * (2.0 * SA - this->GetInput()->GetPixel(idx)));
      }
    }

    // Mirror right (equation 3b in [Ohnesorge et al, Med Phys, 2000])
    RegionType rightRegion = inputRegion;
    rightRegion.SetIndex(0, inputRegion.GetIndex(0) + inputRegion.GetSize(0));
    rightRegion.SetSize(0, next);
    itk::ImageRegionIteratorWithIndex<FFTInputImageType> itRight(paddedImage, rightRegion);
    while (!itRight.IsAtEnd())
    {
      iidx = itRight.GetIndex();
      iidx[0] = rightRegion.GetIndex(0) - 1;
      TFFTPrecision SE = this->GetInput()->GetPixel(iidx);
      for (unsigned int i = 0; i < rightRegion.GetSize(0); i++, ++itRight)
      {
        idx = itRight.GetIndex();
        rightIdx = inputRegion.GetIndex(0) + inputRegion.GetSize(0) - 1;
        borderDist = idx[0] - rightIdx;
        idx[0] = rightIdx - borderDist;
        itRight.Set(m_TruncationMirrorWeights[borderDist] * (2.0 * SE - this->GetInput()->GetPixel(idx)));
      }
    }
  }

  // Copy central part
  itk::ImageRegionConstIterator<InputImageType> itS(this->GetInput(), inputRegion);
  itk::ImageRegionIterator<FFTInputImageType>   itD(paddedImage, inputRegion);
  itS.GoToBegin();
  itD.GoToBegin();
  while (!itS.IsAtEnd())
  {
    itD.Set(itS.Get());
    ++itS;
    ++itD;
  }

  return paddedImage;
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
typename FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::RegionType
FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::GetPaddedImageRegion(
  const RegionType & inputRegion)
{
  RegionType paddedRegion = inputRegion;

  // Set x padding
  typename SizeType::SizeValueType xPaddedSize = m_ZeroPadFactors[0] * inputRegion.GetSize(0);
  while (GreatestPrimeFactor(xPaddedSize) > m_GreatestPrimeFactor)
    xPaddedSize++;
  paddedRegion.SetSize(0, xPaddedSize);
  long zeroext = ((long)xPaddedSize - (long)inputRegion.GetSize(0)) / 2;
  paddedRegion.SetIndex(0, inputRegion.GetIndex(0) - zeroext);

  // Set y padding. Padding along Y is only required if
  // - there is some windowing in the Y direction
  // - the DFT requires the size to be the product of given prime factors
  typename SizeType::SizeValueType yPaddedSize = inputRegion.GetSize(1);
  if (m_KernelDimension == 2)
    yPaddedSize *= m_ZeroPadFactors[1];
  while (GreatestPrimeFactor(yPaddedSize) > m_GreatestPrimeFactor)
    yPaddedSize++;
  paddedRegion.SetSize(1, yPaddedSize);
  // TODO what's that in ScatterGlare long zeroexty = ((long)yPaddedSize - (long)inputRegion.GetSize(1)) / 2;
  paddedRegion.SetIndex(1, inputRegion.GetIndex(1));

  return paddedRegion;
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::PrintSelf(std::ostream & os,
                                                                                          itk::Indent    indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "GreatestPrimeFactor: " << m_GreatestPrimeFactor << std::endl;
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
bool
FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::IsPrime(int n) const
{
  int last = (int)std::sqrt(double(n));

  for (int x = 2; x <= last; x++)
    if (n % x == 0)
      return false;
  return true;
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
int
FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::GreatestPrimeFactor(int n) const
{
  int v = 2;

  while (v <= n)
    if (n % v == 0 && IsPrime(v))
      n /= v;
    else
      v += 1;
  return v;
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
FFTProjectionsConvolutionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::UpdateTruncationMirrorWeights()
{
  const unsigned int next = this->GetTruncationCorrectionExtent();

  if ((unsigned int)m_TruncationMirrorWeights.size() != next)
  {
    m_TruncationMirrorWeights.resize(next + 1);
    for (unsigned int i = 0; i < next + 1; i++)
      m_TruncationMirrorWeights[i] = pow(sin((next - i) * itk::Math::pi / (2 * next - 2)), 0.75);
  }
}

} // end namespace rtk
#endif
