#include "rtkTest.h"
#include "rtkConstantImageSource.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkZengForwardProjectionImageFilter.h"
#ifdef USE_CUDA
#  include "rtkCudaZengForwardProjectionImageFilter.h"
#endif
#include <cmath>
#include <algorithm>
#include <itkNumericTraits.h>
#include <itkImageRegionConstIteratorWithIndex.h>

/**
 * \file rtkzengpsftest.cxx
 *
 * \brief Functional test of the PSF of the Zeng forward projector
 *
 * A point source (one voxel) is forward projected with one parallel
 * projection at a distance d of the detector. The standard deviation of the
 * projection along its first axis must be SigmaZero + Alpha * d, and along
 * its second axis SigmaZeroY + AlphaY * d, SigmaZeroY and AlphaY being the
 * ones of the first axis when they are not set. A negative SigmaZeroY
 * is a value: the standard deviation is then clamped to 0 near the detector,
 * max(0, SigmaZeroY + AlphaY * d), for the volume going up to the detector
 * and for a point nearer to the detector than -SigmaZeroY / AlphaY (no blur).
 *
 * \author David Sarrut
 */

namespace
{
struct PSFCase
{
  const char * name;
  double       distance;
  double       sigmaZero;
  double       alpha;
  double       sigmaZeroY; // itk::NumericTraits<double>::max(): not set
  double       alphaY;     // itk::NumericTraits<double>::max(): not set
};
} // namespace

int
rtkzengpsftest(int, char *[])
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;
#ifdef USE_CUDA
  using ImageType = itk::CudaImage<OutputPixelType, Dimension>;
  using ProjectorType = rtk::CudaZengForwardProjectionImageFilter;
#else
  using ImageType = itk::Image<OutputPixelType, Dimension>;
  using ProjectorType = rtk::ZengForwardProjectionImageFilter<ImageType, ImageType>;
#endif
  using ConstantImageSourceType = rtk::ConstantImageSource<ImageType>;

  // Point source: the central voxel of a volume of 81^3 voxels of 1 mm
  constexpr unsigned int n = 81;
  auto                   volumeSource = ConstantImageSourceType::New();
  volumeSource->SetOrigin(itk::MakePoint(-40., -40., -40.));
  volumeSource->SetSpacing(itk::MakeVector(1., 1., 1.));
  volumeSource->SetSize(itk::MakeSize(n, n, n));
  volumeSource->SetConstant(0.);
  auto point = rtk::DrawEllipsoidImageFilter<ImageType, ImageType>::New();
  point->SetInput(volumeSource->GetOutput());
  point->SetCenter(itk::MakePoint(0., 0., 0.));
  point->SetAxis(itk::MakeVector(0.5, 0.5, 0.5));
  point->SetDensity(1.);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(point->Update());

  auto projectionSource = ConstantImageSourceType::New();
  projectionSource->SetOrigin(itk::MakePoint(-40., -40., 0.));
  projectionSource->SetSpacing(itk::MakeVector(1., 1., 1.));
  projectionSource->SetSize(itk::MakeSize(n, n, 1));
  projectionSource->SetConstant(0.);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(projectionSource->Update());

  const double     notSet = itk::NumericTraits<double>::max();
  constexpr double s0 = 1.15, a = 0.0272, s0y = 1.40, ay = 0.0369, s0yNegative = -0.25;
  const PSFCase    cases[] = {
    { "isotropic (SigmaZeroY, AlphaY not set)", 60., s0, a, notSet, notSet },
    { "anisotropic", 60., s0, a, s0y, ay },
    { "anisotropic", 160., s0, a, s0y, ay },
    { "negative SigmaZeroY", 110., s0, a, s0yNegative, ay },
    { "negative SigmaZeroY, volume up to the detector", 40., s0, a, s0yNegative, ay },
    { "negative SigmaZeroY, point nearer than -SigmaZeroY / AlphaY", 3., s0, a, s0yNegative, ay },
  };
  constexpr double tolerance = 0.02; // mm, on the standard deviation

  for (const auto & c : cases)
  {
    auto geometry = rtk::ThreeDCircularProjectionGeometry::New();
    geometry->AddProjection(c.distance, 0., 0.);

    auto zfw = ProjectorType::New();
    zfw->InPlaceOff();
    zfw->SetInput(0, projectionSource->GetOutput());
    zfw->SetInput(1, point->GetOutput());
    zfw->SetGeometry(geometry);
    zfw->SetSigmaZero(c.sigmaZero);
    zfw->SetAlpha(c.alpha);
    if (c.sigmaZeroY != notSet)
      zfw->SetSigmaZeroY(c.sigmaZeroY);
    if (c.alphaY != notSet)
      zfw->SetAlphaY(c.alphaY);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(zfw->Update());

    // Sum and standard deviation of the projection along its two axes
    double sum = 0.;
    double m1[2] = { 0., 0. };
    double m2[2] = { 0., 0. };
    for (itk::ImageRegionConstIteratorWithIndex<ImageType> it(zfw->GetOutput(),
                                                              zfw->GetOutput()->GetLargestPossibleRegion());
         !it.IsAtEnd();
         ++it)
    {
      const double v = it.Get();
      sum += v;
      for (unsigned int d = 0; d < 2; d++)
      {
        const double x = it.GetIndex()[d];
        m1[d] += v * x;
        m2[d] += v * x * x;
      }
    }

    // Expected standard deviations (pixels of 1 mm)
    const double sigmaZeroY = c.sigmaZeroY == notSet ? c.sigmaZero : c.sigmaZeroY;
    const double alphaY = c.alphaY == notSet ? c.alpha : c.alphaY;
    const double expected[2] = { std::max(0., c.sigmaZero + c.alpha * c.distance),
                                 std::max(0., sigmaZeroY + alphaY * c.distance) };

    std::cout << "\n\n****** " << c.name << ", distance " << c.distance << " mm ******" << std::endl;
    bool ok = std::abs(sum - 1.) < 1e-3;
    std::cout << "sum of the projection " << sum << " (expected 1)" << std::endl;
    for (unsigned int d = 0; d < 2; d++)
    {
      const double mean = m1[d] / sum;
      const double sigma = std::sqrt(m2[d] / sum - mean * mean);
      const bool   okAxis = std::abs(sigma - expected[d]) < tolerance;
      ok = ok && okAxis;
      std::cout << "axis " << d << ": sigma " << sigma << " mm, expected " << expected[d];
      std::cout << (okAxis ? "" : "  <-- WRONG") << std::endl;
    }
    if (!ok)
    {
      std::cerr << "Test FAILED! " << c.name << ", distance " << c.distance << " mm" << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "\n\nTest PASSED! " << std::endl;
  }

  return EXIT_SUCCESS;
}
