#include <itkImageRegionConstIterator.h>
#include <itkStreamingImageFilter.h>

#include "rtkConfiguration.h"
#include "rtkTestConfiguration.h"
#include "rtkTest.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"

#ifdef USE_CUDA
#  include "rtkCudaIterativeFDKConeBeamReconstructionFilter.h"
#else
#  include "rtkIterativeFDKConeBeamReconstructionFilter.h"
#endif

/**
 * \file rtkiterativefdktest.cxx
 *
 * \brief Functional test for iterative FDK reconstruction
 *
 * This test generates the projections of a simulated Shepp-Logan phantom.
 * A CT image is reconstructed from the set of generated projection images
 * using the iterative FDK algorithm and the reconstructed CT image is compared
 * to the expected results which is analytically computed.
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
  auto size = itk::MakeSize(32, 32, 32);
  auto spacing = itk::MakeVector(8., 8., 8.);
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
  size = itk::MakeSize(32, 32, NumberOfProjectionImages);
  spacing = itk::MakeVector(32., 32., 32.);
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
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

  // Shepp Logan projections filter
  using SLPType = rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType>;
  auto slp = SLPType::New();
  slp->SetInput(projectionsSource->GetOutput());
  slp->SetGeometry(geometry);
  slp->SetPhantomScale(116);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(slp->Update());

  // Create a reference object (in this case a 3D phantom reference).
  using DSLType = rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType>;
  auto dsl = DSLType::New();
  dsl->SetInput(tomographySource->GetOutput());
  dsl->SetPhantomScale(116);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update())

  // FDK reconstruction filtering
#ifdef USE_CUDA
  using FDKType = rtk::CudaIterativeFDKConeBeamReconstructionFilter;
#else
  using FDKType = rtk::IterativeFDKConeBeamReconstructionFilter<OutputImageType>;
#endif
  auto ifdk = FDKType::New();
  ifdk->SetInput(0, tomographySource->GetOutput());
  ifdk->SetInput(1, slp->GetOutput());
  ifdk->SetGeometry(geometry);
  ifdk->SetNumberOfIterations(3);
#ifdef USE_CUDA
  ifdk->SetForwardProjectionFilter(FDKType::FP_CUDARAYCAST);
#else
  ifdk->SetForwardProjectionFilter(FDKType::FP_JOSEPH);
#endif
  TRY_AND_EXIT_ON_ITK_EXCEPTION(ifdk->Update());

  // FOV
  using FOVFilterType = rtk::FieldOfViewImageFilter<OutputImageType, OutputImageType>;
  auto fov = FOVFilterType::New();
  fov->SetInput(0, ifdk->GetOutput());
  fov->SetProjectionsStack(slp->GetOutput());
  fov->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(fov->Update());

  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput(), 0.027, 27, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
