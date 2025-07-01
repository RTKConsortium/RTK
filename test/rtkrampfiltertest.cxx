#include <itkImageRegionConstIterator.h>

#include "rtkTestConfiguration.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkAdditiveGaussianNoiseImageFilter.h"
#include "rtkTest.h"

#ifdef USE_CUDA
#  include "rtkCudaFDKConeBeamReconstructionFilter.h"
#else
#  include "rtkFDKConeBeamReconstructionFilter.h"
#endif

/**
 * \file rtkrampfiltertest.cxx
 *
 * \brief Functional test for the ramp filter of the FDK reconstruction.
 *
 * This test generates the projections of a simulated Shepp-Logan phantom in
 * different reconstruction scenarios (noise, truncation).
 * CT images are reconstructed from each set of projection images using the
 * FDK algorithm with different configuration of the ramp filter in order to
 * reduce the possible artifacts. The generated results are compared to the
 * expected results (analytical calculation).
 *
 * \author Simon Rit
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;
#ifdef USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif
#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 180;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto tomographySource = ConstantImageSourceType::New();
  auto origin = itk::MakePoint(-127., -127., -127.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(2, 2, 2);
  auto spacing = itk::MakeVector(254., 254., 254.);
#else
  auto size = itk::MakeSize(128, 128, 128);
  auto spacing = itk::MakeVector(2., 2., 2.);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  auto projectionsSource = ConstantImageSourceType::New();
  origin = itk::MakePoint(-254., -254., -254.);
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(2, 2, NumberOfProjectionImages);
  spacing = itk::MakeVector(508., 508., 508.);
#else
  size = itk::MakeSize(128, 128, NumberOfProjectionImages);
  spacing = itk::MakeVector(4., 4., 4.);
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  // Geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto geometry = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);

  // Shepp Logan projections filter
  using SLPType = rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType>;
  auto slp = SLPType::New();
  slp->SetInput(projectionsSource->GetOutput());
  slp->SetGeometry(geometry);

  std::cout << "\n\n****** Test 1: add noise and test Hann window ******" << std::endl;

  // Add noise
  using NIFType = rtk::AdditiveGaussianNoiseImageFilter<OutputImageType>;
  auto noisy = NIFType::New();
  noisy->SetInput(slp->GetOutput());
  noisy->SetMean(0.0);
  noisy->SetStandardDeviation(1.);

  // Create a reference object (in this case a 3D phantom reference).
  using DSLType = rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType>;
  auto dsl = DSLType::New();
  dsl->SetInput(tomographySource->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update());

  // FDK reconstruction filtering
#ifdef USE_CUDA
  using FDKType = rtk::CudaFDKConeBeamReconstructionFilter;
#else
  using FDKType = rtk::FDKConeBeamReconstructionFilter<OutputImageType>;
#endif
  auto feldkamp = FDKType::New();
  feldkamp->SetInput(0, tomographySource->GetOutput());
  feldkamp->SetInput(1, noisy->GetOutput());
  feldkamp->SetGeometry(geometry);
  feldkamp->GetRampFilter()->SetHannCutFrequency(0.8);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkamp->Update());

  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.72, 22.4, 2.);

  std::cout << "\n\n****** Test 1.5: add noise and test HannY window ******" << std::endl;
  feldkamp->GetRampFilter()->SetHannCutFrequencyY(0.8);
  feldkamp->Modified();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkamp->Update());
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.74, 21.8, 2.);

  std::cout << "\n\n****** Test 2: smaller detector and test data padding for truncation ******" << std::endl;

  size[0] = 114;
  projectionsSource->SetSize(size);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(slp->UpdateLargestPossibleRegion());

  auto feldkampCropped = FDKType::New();
  feldkampCropped->SetInput(0, tomographySource->GetOutput());
  feldkampCropped->SetInput(1, slp->GetOutput());
  feldkampCropped->SetGeometry(geometry);
  feldkampCropped->GetRampFilter()->SetTruncationCorrection(0.1);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkampCropped->Update());

  CheckImageQuality<OutputImageType>(feldkampCropped->GetOutput(), dsl->GetOutput(), 0.12, 20.8, 2.);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
