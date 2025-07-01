
#ifdef RAMP_FILTER_TEST_WITHOUT_FFTW
#  include "rtkConfiguration.h"
#  include <itkImageToImageFilter.h>
#  if defined(ITK_USE_FFTWF)
#    undef ITK_USE_FFTWF
#  endif
#  if defined(ITK_USE_FFTWD)
#    undef ITK_USE_FFTWD
#  endif
#  if defined(USE_FFTWF)
#    undef USE_FFTWF
#  endif
#  if defined(USE_FFTWD)
#    undef USE_FFTWD
#  endif
#endif

#include "rtkTest.h"
#include "rtkTestConfiguration.h"
#include "rtkMacro.h"

#include <cmath>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#ifdef USE_CUDA
#  include "rtkCudaScatterGlareCorrectionImageFilter.h"
#else
#  include "rtkScatterGlareCorrectionImageFilter.h"
#endif

#include "rtkTestConfiguration.h"

/**
 * \file rtkscatterglarefiltertest.cxx
 *
 * \brief Functional test for the scatter glare correction filter
 *
 * \author Sebastien Brousmiche
 */

constexpr unsigned int Dimension = 3;
#ifdef USE_CUDA
using ImageType = itk::CudaImage<float, Dimension>;
#else
using ImageType = itk::Image<float, Dimension>;
#endif

constexpr float spikeValue = 12.341;

ImageType::Pointer
createInputImage(const std::vector<float> & coef)
{
  auto                  spacing = itk::MakeVector(0.296, 0.296, 1.);
  auto                  size = itk::MakeSize(650, 700, 1);
  ImageType::RegionType region;
  region.SetSize(size);

  auto inputI = ImageType::New();
  inputI->SetRegions(region);
  inputI->SetSpacing(spacing);
  inputI->Allocate();
  inputI->FillBuffer(1.0f);

  float a3 = coef[0];
  float b3 = coef[1];
  float b3sq = b3 * b3;
  float dx = spacing[0];
  float dy = spacing[1];

  itk::ImageRegionIteratorWithIndex<ImageType> itK(inputI, inputI->GetLargestPossibleRegion());
  itK.GoToBegin();
  ImageType::IndexType idx;
  while (!itK.IsAtEnd())
  {
    idx = itK.GetIndex();
    float xx = (float)idx[0] - (float)size[0] / 2.0f;
    float yy = (float)idx[1] - (float)size[1] / 2.0f;
    float rr2 = (xx * xx + yy * yy);
    float g = (a3 * dx * dy / (2.0f * itk::Math::pi * b3sq)) * 1.0f / std::pow((1.0f + rr2 / b3sq), 1.5f);
    if ((2 * idx[0] == (ImageType::IndexValueType)size[0]) && ((2 * idx[1] == (ImageType::IndexValueType)size[1])))
    {
      g += (1 - a3);
    }
    g = spikeValue * g; // The image is a spike at the central pixel convolved with the scatter PSF
    itK.Set(g);
    ++itK;
  }
  return inputI;
}

int
main(int, char **)
{
#ifdef USE_CUDA
  using ScatterCorrectionType = rtk::CudaScatterGlareCorrectionImageFilter;
#else
  using ScatterCorrectionType = rtk::ScatterGlareCorrectionImageFilter<ImageType, ImageType, float>;
#endif
  auto SFilter = ScatterCorrectionType::New();

  std::vector<float> coef;
  coef.push_back(0.0787f);
  coef.push_back(106.244f);

  SFilter->SetTruncationCorrection(0.5);
  SFilter->SetCoefficients(coef);

  ImageType::Pointer testImage = createInputImage(coef);
  SFilter->SetInput(testImage);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(SFilter->Update())

  ImageType::Pointer                       outputI = SFilter->GetOutput();
  ImageType::SizeType                      size = outputI->GetLargestPossibleRegion().GetSize();
  itk::ImageRegionConstIterator<ImageType> itO(outputI, outputI->GetLargestPossibleRegion());
  itO.GoToBegin();

  ImageType::IndexType idx;
  float                sumBng = 0.0f;
  float                spikeValueOut = 0.0f;
  while (!itO.IsAtEnd())
  {
    idx = itO.GetIndex();
    if ((2 * idx[0] == (ImageType::IndexValueType)size[0]) && ((2 * idx[1] == (ImageType::IndexValueType)size[1])))
    {
      spikeValueOut += itO.Get();
    }
    else
    {
      sumBng += itO.Get();
    }
    ++itO;
  }

  if (!((itk::Math::abs(spikeValueOut - spikeValue) < 1e-2) && (itk::Math::abs(sumBng) < 1e-2)))
  {
    std::cerr << "Test Failed! " << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
