#include <itkRandomImageSource.h>
#include <itkImageRegionIterator.h>

#include "rtkConstantImageSource.h"
#include "rtkTestConfiguration.h"
#include "rtkConjugateGradientImageFilter.h"
#include "rtkDivergenceOfGradientConjugateGradientOperator.h"
#include "rtkMacro.h"
#include "rtkForwardDifferenceGradientImageFilter.h"
#include "rtkBackwardDifferenceDivergenceImageFilter.h"


template <class TImage1, class TImage2>
#if FAST_TESTS_NO_CHECKS
void
CheckImageQuality(typename TImage1::Pointer itkNotUsed(recon), typename TImage2::Pointer itkNotUsed(ref))
{}
#else
void
CheckImageQuality(typename TImage1::Pointer recon, typename TImage2::Pointer ref)
{
  using ImageIteratorType1 = itk::ImageRegionConstIterator<TImage1>;
  using ImageIteratorType2 = itk::ImageRegionConstIterator<TImage2>;
  ImageIteratorType1 itTest(recon, recon->GetBufferedRegion());
  ImageIteratorType2 itRef(ref, ref->GetBufferedRegion());

  // Compute the mean of the reference image (which cannot be recovered by inverting the gradient)
  double mean = 0;
  while (!itRef.IsAtEnd())
  {
    mean += itRef.Get();
    ++itRef;
  }
  mean /= ref->GetBufferedRegion().GetNumberOfPixels();

  using ErrorType = double;
  ErrorType TestError = 0.;
  ErrorType EnerError = 0.;

  itTest.GoToBegin();
  itRef.GoToBegin();

  while (!itRef.IsAtEnd())
  {
    typename TImage1::PixelType TestVal = itTest.Get();
    typename TImage2::PixelType RefVal = itRef.Get() - mean;

    if (TestVal != RefVal)
    {
      TestError += itk::Math::abs(RefVal - TestVal);
      EnerError += std::pow(ErrorType(RefVal - TestVal), 2.);
    }
    ++itTest;
    ++itRef;
  }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError / recon->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError / ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "MSE = " << MSE << std::endl;
  // PSNR
  ErrorType PSNR = 20 * log10(255.0) - 10 * log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (255.0 - ErrorPerPixel) / 255.0;
  std::cout << "QI = " << QI << std::endl;

  // Checking results
  if (ErrorPerPixel > 1.28)
  {
    std::cerr << "Test Failed, Error per pixel not valid! " << ErrorPerPixel << " instead of 1.28" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (PSNR < 44.)
  {
    std::cerr << "Test Failed, PSNR not valid! " << PSNR << " instead of 44" << std::endl;
    exit(EXIT_FAILURE);
  }
}
#endif

/**
 * \file rtkconjugategradienttest.cxx
 *
 * \brief Attempts to recover an image from its gradient using
 * conjugate gradient optimization
 *
 * This test generates a random volume and computes its gradient.
 * It attempts to recover the random volume (except for the DC component)
 * from its gradient. The optimizer attempts to solve
 * find f such that || grad(f) - g ||_2^2 is minimal
 * It searches for the zeros of the gradient of this quantity, i.e. for
 * f such that div( grad(f) - g ) = 0
 * which is equivalent to
 * div(grad(f)) = div(g)
 * Expressed in the canonical form AX=B, this means :
 * A = div(grad(.))
 * X = f
 * B = div(g)
 *
 *
 * \author Cyril Mory
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;

  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  // Random and constant image sources
  using RandomImageSourceType = itk::RandomImageSource<OutputImageType>;
  RandomImageSourceType::Pointer randomVolumeSource = RandomImageSourceType::New();
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::Pointer constantVolumeSource = ConstantImageSourceType::New();

  // Image meta data
  RandomImageSourceType::PointType   origin;
  RandomImageSourceType::SizeType    size;
  RandomImageSourceType::SpacingType spacing;

  // Volume metadata
  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = 2;
  spacing[0] = 252.;
  spacing[1] = 252.;
  spacing[2] = 252.;
#else
  size[0] = 64;
  size[1] = 64;
  size[2] = 64;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
#endif
  randomVolumeSource->SetOrigin(origin);
  randomVolumeSource->SetSpacing(spacing);
  randomVolumeSource->SetSize(size);
  randomVolumeSource->SetMin(0.);
  randomVolumeSource->SetMax(1.);
  randomVolumeSource->SetNumberOfWorkUnits(2); // With 1, it's deterministic

  constantVolumeSource->SetOrigin(origin);
  constantVolumeSource->SetSpacing(spacing);
  constantVolumeSource->SetSize(size);
  constantVolumeSource->SetConstant(0.);

  // Update the sources
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomVolumeSource->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantVolumeSource->Update());

  // Create and set the gradient and divergence filters
  using GradientFilterType = rtk::ForwardDifferenceGradientImageFilter<OutputImageType>;
  GradientFilterType::Pointer gradientFilter = GradientFilterType::New();
  gradientFilter->SetInput(randomVolumeSource->GetOutput());
  bool dimsProcessed[Dimension];
  for (bool & dimProcessed : dimsProcessed)
  {
    dimProcessed = true;
  }
  gradientFilter->SetDimensionsProcessed(dimsProcessed);
  using DivergenceFilterType = rtk::BackwardDifferenceDivergenceImageFilter<GradientFilterType::OutputImageType>;
  DivergenceFilterType::Pointer divergenceFilter = DivergenceFilterType::New();
  divergenceFilter->SetInput(gradientFilter->GetOutput());

  // Update the gradient and divergence filters
  TRY_AND_EXIT_ON_ITK_EXCEPTION(divergenceFilter->Update());

  // Create and set the conjugate gradient optimizer
  // It uses the operator DivergenceOfGradientConjugateGradientOperator
  using CGFilterType = rtk::ConjugateGradientImageFilter<OutputImageType>;
  CGFilterType::Pointer cg = CGFilterType::New();

  using CGoperatorType = rtk::DivergenceOfGradientConjugateGradientOperator<OutputImageType>;
  CGoperatorType::Pointer cg_op = CGoperatorType::New();
  cg->SetA(cg_op.GetPointer());
  cg->SetX(constantVolumeSource->GetOutput());
  cg->SetB(divergenceFilter->GetOutput());
  cg->SetNumberOfIterations(30);

  // Update the conjugate gradient filter
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cg->Update());

  CheckImageQuality<OutputImageType, OutputImageType>(cg->GetOutput(), randomVolumeSource->GetOutput());

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
