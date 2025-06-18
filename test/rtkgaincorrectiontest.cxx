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

#include "rtkTest.h"
#include "rtkTestConfiguration.h"
#include "rtkMacro.h"

#include <cmath>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#ifdef USE_CUDA
#  include "rtkCudaPolynomialGainCorrectionImageFilter.h"
#else
#  include "rtkPolynomialGainCorrectionImageFilter.h"
#endif

#include "rtkTestConfiguration.h"

/**
 * \file rtkgaincorrectiontest.cxx
 *
 * \brief Functional test for the polynomial gain correction filter
 *
 * \author Sebastien Brousmiche
 */

constexpr unsigned int Dimension = 3;
#ifdef USE_CUDA
using InputImageType = itk::CudaImage<unsigned short, Dimension>;
using OutputImageType = itk::CudaImage<float, Dimension>;
#else
using InputImageType = itk::Image<unsigned short, Dimension>;
using OutputImageType = itk::Image<float, Dimension>;
#endif

constexpr int modelOrder = 3;
constexpr int sizeI = 100;

InputImageType::Pointer
createDarkImage()
{
  auto size = itk::MakeSize(sizeI, sizeI, 1);
  auto spacing = itk::MakeVector(1.f, 1.f, 1.f);

  InputImageType::RegionType region;
  region.SetSize(size);

  InputImageType::Pointer darkImage = InputImageType::New();
  darkImage->SetRegions(region);
  darkImage->SetSpacing(spacing);
  darkImage->Allocate();
  darkImage->FillBuffer(0.0f);

  float orig_x = -static_cast<float>(size[0] - 1) / 2.f;
  float orig_y = -static_cast<float>(size[1] - 1) / 2.f;

  itk::ImageRegionIteratorWithIndex<InputImageType> itIn(darkImage, darkImage->GetLargestPossibleRegion());
  itIn.GoToBegin();
  while (!itIn.IsAtEnd())
  {
    InputImageType::IndexType idx = itIn.GetIndex();
    float                     xx = static_cast<float>(idx[0]) + orig_x;
    float                     yy = static_cast<float>(idx[1]) + orig_y;
    float                     rr = std::sqrt(xx * xx + yy * yy);
    itIn.Set(static_cast<unsigned short>(rr));
    ++itIn;
  }

  return darkImage;
}

OutputImageType::Pointer
createGainImage()
{
  auto size = itk::MakeSize(sizeI, sizeI, modelOrder);
  auto spacing = itk::MakeVector(1.f, 1.f, 1.f);

  OutputImageType::RegionType region;
  region.SetSize(size);

  OutputImageType::Pointer gainImage = OutputImageType::New();
  gainImage->SetRegions(region);
  gainImage->SetSpacing(spacing);
  gainImage->Allocate();
  gainImage->FillBuffer(0.0f);

  itk::ImageRegionIteratorWithIndex<OutputImageType> itIn(gainImage, gainImage->GetLargestPossibleRegion());
  itIn.GoToBegin();
  while (!itIn.IsAtEnd())
  {
    InputImageType::IndexType idx = itIn.GetIndex();
    float                     value = 0.f;
    if (idx[2] == 0)
      value = 1.2f;
    else if (idx[2] == 1)
      value = 0.016f;
    else if (idx[2] == 3)
      value = -0.003f;

    itIn.Set(value);
    ++itIn;
  }

  return gainImage;
}

InputImageType::Pointer
createInputImage()
{
  auto size = itk::MakeSize(sizeI, sizeI, 1);
  auto spacing = itk::MakeVector(1.f, 1.f, 1.f);

  InputImageType::RegionType region;
  region.SetSize(size);

  InputImageType::Pointer inputImage = InputImageType::New();
  inputImage->SetRegions(region);
  inputImage->SetSpacing(spacing);
  inputImage->Allocate();
  inputImage->FillBuffer(32000.0f);

  return inputImage;
}

OutputImageType::Pointer
generateExpectedOutput(InputImageType::Pointer  inputImage,
                       float                    K,
                       InputImageType::Pointer  darkImage,
                       OutputImageType::Pointer gainImage)
{
  auto size = itk::MakeSize(sizeI, sizeI, 1);
  auto spacing = itk::MakeVector(1.f, 1.f, 1.f);

  OutputImageType::RegionType region;
  region.SetSize(size);

  OutputImageType::Pointer expectedOutput = OutputImageType::New();
  expectedOutput->SetRegions(region);
  expectedOutput->SetSpacing(spacing);
  expectedOutput->Allocate();
  expectedOutput->FillBuffer(0.0f);

  itk::ImageRegionConstIterator<InputImageType> itIn(inputImage, inputImage->GetLargestPossibleRegion());
  itk::ImageRegionIterator<OutputImageType>     itOut(expectedOutput, expectedOutput->GetLargestPossibleRegion());

  InputImageType::RegionType darkRegion = inputImage->GetLargestPossibleRegion();
  darkRegion.SetSize(2, 1);
  darkRegion.SetIndex(2, 0);
  itk::ImageRegionConstIterator<InputImageType> itDark(darkImage, darkRegion);

  // Get gain map buffer
  const float * gainBuffer = gainImage->GetBufferPointer();

  itIn.GoToBegin();
  itOut.GoToBegin();
  itDark.GoToBegin();

  float sizeI2 = sizeI * sizeI;
  for (int j = 0; j < sizeI; ++j)
  {
    for (int i = 0; i < sizeI; ++i)
    {
      int darkx = static_cast<int>(itDark.Get());
      int px = static_cast<int>(itIn.Get()) - darkx;
      px = (px >= 0) ? px : 0;

      float correctedValue = 0.f;
      auto  powValue = static_cast<float>(px);
      for (int m = 0; m < modelOrder; ++m)
      {
        int   gainidx = m * sizeI2 + j * sizeI + i;
        float Aij = gainBuffer[gainidx];
        correctedValue += Aij * powValue;
        powValue *= powValue;
      }
      correctedValue *= K;

      itOut.Set(static_cast<float>(correctedValue));

      ++itIn;
      ++itOut;
      ++itDark;
    }
  }

  return expectedOutput;
}


int
main(int, char **)
{
  const float K = 0.5f;
#ifdef USE_CUDA
  using GainCorrectionType = rtk::CudaPolynomialGainCorrectionImageFilter;
#else
  using GainCorrectionType = rtk::PolynomialGainCorrectionImageFilter<InputImageType, OutputImageType>;
#endif
  GainCorrectionType::Pointer gainfilter = GainCorrectionType::New();

  // Set filter inputs
  InputImageType::Pointer darkImage = createDarkImage();
  gainfilter->SetDarkImage(darkImage);

  OutputImageType::Pointer gainImage = createGainImage();
  gainfilter->SetGainCoefficients(gainImage);

  gainfilter->SetK(K);

  InputImageType::Pointer testImage = createInputImage();
  gainfilter->SetInput(testImage);

  // Apply correction
  TRY_AND_EXIT_ON_ITK_EXCEPTION(gainfilter->Update())
  OutputImageType::Pointer outputImage = gainfilter->GetOutput();

  // Generate expected output
  OutputImageType::Pointer expectedOutput = generateExpectedOutput(testImage, K, darkImage, gainImage);

  // Compare
  itk::ImageRegionConstIterator<OutputImageType> itExp(expectedOutput, expectedOutput->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<OutputImageType> itOut(outputImage, outputImage->GetLargestPossibleRegion());

  itExp.GoToBegin();
  itOut.GoToBegin();
  float diffValue = 0.f;
  while (!itExp.IsAtEnd())
  {
    diffValue += itk::Math::abs(itExp.Get() - itOut.Get());
    ++itExp;
    ++itOut;
  }
  diffValue /= static_cast<float>(sizeI * sizeI);
  std::cout << diffValue << std::endl;
  if (!(diffValue < 1.f))
  {
    std::cerr << "Test Failed! " << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
