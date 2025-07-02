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

#ifndef rtkScatterGlareCorrectionImageFilter_hxx
#define rtkScatterGlareCorrectionImageFilter_hxx


// Use local RTK FFTW files taken from Gaëtan Lehmann's code for
// thread safety: https://hdl.handle.net/10380/3154
#include <itkRealToHalfHermitianForwardFFTImageFilter.h>
#include <itkHalfHermitianToRealInverseFFTImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkDivideImageFilter.h>

namespace rtk
{

template <class TInputImage, class TOutputImage, class TFFTPrecision>
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::ScatterGlareCorrectionImageFilter()
{
  this->m_KernelDimension = 2;
}

template <class TInputImage, class TOutputImage, class TFFTPrecision>
void
ScatterGlareCorrectionImageFilter<TInputImage, TOutputImage, TFFTPrecision>::UpdateFFTProjectionsConvolutionKernel(
  const SizeType size)
{
  if (m_Coefficients.size() != 2)
  {
    itkGenericExceptionMacro(<< "Expecting 2 coefficients in m_Coefficients)");
  }
  double                dx = this->GetInput()->GetSpacing()[0];
  double                dy = this->GetInput()->GetSpacing()[1];
  CoefficientVectorType coeffs = m_Coefficients;
  coeffs.push_back(dx);
  coeffs.push_back(dy);
  coeffs.push_back(size[0]);
  coeffs.push_back(size[1]);
  if (coeffs == m_PreviousCoefficients)
    return; // Up-to-date
  m_PreviousCoefficients = coeffs;

  FFTInputImagePointer kernel = FFTInputImageType::New();
  kernel->SetRegions(size);
  kernel->Allocate();

  double a3 = m_Coefficients[0];
  double b3 = m_Coefficients[1];
  double b3sq = b3 * b3;
  double halfXSz = size[0] / 2.;
  double halfYSz = size[1] / 2.;

  itk::ImageRegionIteratorWithIndex<FFTInputImageType> itK(kernel, kernel->GetLargestPossibleRegion());
  itK.GoToBegin();

  // Central value
  double g = (1 - a3) + a3 * dx * dy / (2. * itk::Math::pi * b3sq);
  itK.Set(g);
  ++itK;

  typename FFTInputImageType::IndexType idx;
  while (!itK.IsAtEnd())
  {
    idx = itK.GetIndex();
    double xx = halfXSz - itk::Math::abs(halfXSz - idx[0]); // Distance to nearest x border
    double yy = halfYSz - itk::Math::abs(halfYSz - idx[1]); // Distance to nearest y border
    double rr2 = (xx * xx + yy * yy);
    g = (a3 * dx * dy / (2. * itk::Math::pi * b3sq)) / std::pow((1. + rr2 / b3sq), 1.5);
    itK.Set(g);
    ++itK;
  }

  // FFT kernel
  auto fftK = itk::RealToHalfHermitianForwardFFTImageFilter<FFTInputImageType, FFTOutputImageType>::New();
  fftK->SetInput(kernel);
  fftK->SetNumberOfWorkUnits(this->GetNumberOfWorkUnits());
  fftK->Update();

  // Inverse
  auto div = itk::DivideImageFilter<FFTOutputImageType, FFTOutputImageType, FFTOutputImageType>::New();
  div->SetConstant1(1.);
  div->SetInput(1, fftK->GetOutput());
  div->SetNumberOfWorkUnits(this->GetNumberOfWorkUnits());
  div->Update();

  this->m_KernelFFT = div->GetOutput();
  this->m_KernelFFT->DisconnectPipeline();
}

} // end namespace rtk
#endif
