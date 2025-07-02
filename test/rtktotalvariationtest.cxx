#include "itkRandomImageSource.h"
#include "math.h"

#include "rtkTotalVariationImageFilter.h"
#include "rtkTotalVariationDenoisingBPDQImageFilter.h"
#include "rtkMacro.h"

template <class TImage>
void
CheckTotalVariation(typename TImage::Pointer before, typename TImage::Pointer after)
{
  auto tv = rtk::TotalVariationImageFilter<TImage>::New();

  double totalVariationBefore = NAN;
  double totalVariationAfter = NAN;

  tv->SetInput(before);
  tv->Update();
  totalVariationBefore = tv->GetTotalVariation();
  std::cout << "Total variation before denoising is " << totalVariationBefore << std::endl;

  tv->SetInput(after);
  tv->Update();
  totalVariationAfter = tv->GetTotalVariation();
  std::cout << "Total variation after denoising is " << totalVariationAfter << std::endl;

  // Checking results
  if (totalVariationBefore / 2 < totalVariationAfter)
  {
    std::cerr << "Test Failed: total variation was not reduced enough" << std::endl;
    exit(EXIT_FAILURE);
  }
}

/**
 * \file rtktotalvariationtest.cxx
 *
 * \brief Tests whether the Total Variation denoising BPDQ filter indeed
 * reduces the total variation of a random image
 *
 * This test generates a random volume and performs TV denoising on this
 * volume. It measures its total variation before and after denoising and
 * compares. Note that the TV denoising filter does not minimize TV alone,
 * but TV + a data attachment term (it computes the proximal operator of TV).
 * Nevertheless, in most cases, it is expected that the output has
 * a lower TV than the input.
 *
 * \author Cyril Mory
 */

int
main(int, char **)
{
  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

#ifdef USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
  using GradientOutputImageType = itk::CudaImage<itk::CovariantVector<OutputPixelType, Dimension>, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  using GradientOutputImageType = itk::Image<itk::CovariantVector<OutputPixelType, Dimension>, Dimension>;
#endif

  // Random image sources
  auto randomVolumeSource = itk::RandomImageSource<OutputImageType>::New();

  // Volume metadata
  auto origin = itk::MakePoint(0., 0., 0.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(64, 64, 1);
  auto spacing = itk::MakeVector(1., 1., 1.);
#else
  auto size = itk::MakeSize(64, 64, 64);
  auto spacing = itk::MakeVector(4., 4., 4.);
#endif
  randomVolumeSource->SetOrigin(origin);
  randomVolumeSource->SetSpacing(spacing);
  randomVolumeSource->SetSize(size);
  randomVolumeSource->SetMin(0.);
  randomVolumeSource->SetMax(1.);
  randomVolumeSource->SetNumberOfWorkUnits(2); // With 1, it's deterministic

  // Update the source
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomVolumeSource->Update());

  // Create and set the TV denoising filter
  auto TVdenoising = rtk::TotalVariationDenoisingBPDQImageFilter<OutputImageType, GradientOutputImageType>::New();
  TVdenoising->SetInput(randomVolumeSource->GetOutput());
  TVdenoising->SetNumberOfIterations(100);
  TVdenoising->SetGamma(0.3);

  bool dimsProcessed[Dimension];
  for (bool & dimProcessed : dimsProcessed)
  {
    dimProcessed = true;
  }
  TVdenoising->SetDimensionsProcessed(dimsProcessed);

  // Update the TV denoising filter
  TRY_AND_EXIT_ON_ITK_EXCEPTION(TVdenoising->Update());

  CheckTotalVariation<OutputImageType>(randomVolumeSource->GetOutput(), TVdenoising->GetOutput());

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
