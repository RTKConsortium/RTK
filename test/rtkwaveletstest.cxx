#include <itkImageRegionConstIterator.h>
#include <itkRandomImageSource.h>

#include "rtkTestConfiguration.h"
#include "rtkDeconstructSoftThresholdReconstructImageFilter.h"
#include "rtkMacro.h"

template <class TImage>
#if FAST_TESTS_NO_CHECKS
void
CheckImageQuality(typename TImage::Pointer itkNotUsed(recon), typename TImage::Pointer itkNotUsed(ref))
{}
#else
void
CheckImageQuality(typename TImage::Pointer recon, typename TImage::Pointer ref)
{
  using ImageIteratorType = itk::ImageRegionConstIterator<TImage>;
  ImageIteratorType itTest(recon, recon->GetBufferedRegion());
  ImageIteratorType itRef(ref, ref->GetBufferedRegion());

  using ErrorType = double;
  ErrorType TestError = 0.;
  ErrorType EnerError = 0.;

  itTest.GoToBegin();
  itRef.GoToBegin();

  while (!itRef.IsAtEnd())
  {
    typename TImage::PixelType TestVal = itTest.Get();
    typename TImage::PixelType RefVal = itRef.Get();
    TestError += itk::Math::abs(RefVal - TestVal);
    EnerError += std::pow(ErrorType(RefVal - TestVal), 2.);
    ++itTest;
    ++itRef;
  }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError / ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError / ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "MSE = " << MSE << std::endl;
  // PSNR
  ErrorType PSNR = 20 * log10(2.0) - 10 * log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (2.0 - ErrorPerPixel) / 2.0;
  std::cout << "QI = " << QI << std::endl;

  // Checking results
  if (ErrorPerPixel > 0.032)
  {
    std::cerr << "Test Failed, Error per pixel not valid! " << ErrorPerPixel << " instead of 0.08" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (PSNR < 28)
  {
    std::cerr << "Test Failed, PSNR not valid! " << PSNR << " instead of 23" << std::endl;
    exit(EXIT_FAILURE);
  }
}
#endif

/**
 * \file rtkwaveletstest.cxx
 *
 * \brief Functional test for wavelets deconstruction / reconstruction
 *
 * This test generates a random image, computes its wavelets deconstruction,
 * reconstructs from it, and compares the results to the original image.
 *
 * \author Cyril Mory
 */

int
main(int, char **)
{
  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  // Random image sources
  using RandomImageSourceType = itk::RandomImageSource<OutputImageType>;
  RandomImageSourceType::Pointer randomVolumeSource = RandomImageSourceType::New();

  // Volume metadata
  auto origin = itk::MakePoint(0., 0., 0.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(8, 8, 8);
  auto spacing = itk::MakeVector(32., 32., 32.);
#else
  auto size = itk::MakeSize(64, 64, 64);
  auto spacing = itk::MakeVector(4., 4., 4.);
#endif
  origin.Fill(0.);
  randomVolumeSource->SetOrigin(origin);
  randomVolumeSource->SetSpacing(spacing);
  randomVolumeSource->SetSize(size);
  randomVolumeSource->SetMin(0.);
  randomVolumeSource->SetMax(1.);
  randomVolumeSource->SetNumberOfWorkUnits(2); // With 1, it's deterministic

  // Update the source
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomVolumeSource->Update());

  // Wavelets deconstruction and reconstruction
  using DeconstructReconstructFilterType = rtk::DeconstructSoftThresholdReconstructImageFilter<OutputImageType>;
  DeconstructReconstructFilterType::Pointer wavelets = DeconstructReconstructFilterType::New();
  wavelets->SetInput(randomVolumeSource->GetOutput());
  wavelets->SetNumberOfLevels(3);
  wavelets->SetOrder(3);
  wavelets->SetThreshold(0);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(wavelets->Update());

  CheckImageQuality<OutputImageType>(wavelets->GetOutput(), randomVolumeSource->GetOutput());

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
