#include <itkImageRegionConstIterator.h>

#include "rtkTest.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkConstantImageSource.h"
#ifdef USE_CUDA
#  include "rtkCudaDisplacedDetectorImageFilter.h"
#else
#  include "rtkDisplacedDetectorImageFilter.h"
#endif

/**
 * \file rtkdisplaceddetectortest.cxx
 *
 * \brief Functional test for classes performing FDK reconstructions with a
 * displaced detector/source
 *
 * This test generates the projections of a simulated Shepp-Logan phantom and
 * different sets of geometries with different displaced detectors and sources.
 * Images are then reconstructed from the generated projection images and
 * compared to the expected results which is analytically computed.
 *
 * \author Simon Rit and Marc Vila
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
  ConstantImageSourceType::Pointer tomographySource = ConstantImageSourceType::New();
  auto                             origin = itk::MakePoint(-127., -127., -127.);
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

  origin = itk::MakePoint(-254., -254., -254.);
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(2, 2, NumberOfProjectionImages);
  spacing = itk::MakeVector(508., 508., 508.);
#else
  size = itk::MakeSize(128, 128, NumberOfProjectionImages);
  spacing = itk::MakeVector(4., 4., 4.);
#endif
  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  // Shepp Logan projections filter
  using SLPType = rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType>;
  SLPType::Pointer slp = SLPType::New();
  slp->SetInput(projectionsSource->GetOutput());

  // Displaced detector weighting
#ifdef USE_CUDA
  using DDFType = rtk::CudaDisplacedDetectorImageFilter;
#else
  using DDFType = rtk::DisplacedDetectorImageFilter<OutputImageType>;
#endif
  DDFType::Pointer ddf = DDFType::New();
  ddf->SetInput(slp->GetOutput());

  // Create a reference object (in this case a 3D phantom reference).
  using DSLType = rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType>;
  DSLType::Pointer dsl = DSLType::New();
  dsl->SetInput(tomographySource->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update());

  // FDK reconstruction filtering
  using FDKCPUType = rtk::FDKConeBeamReconstructionFilter<OutputImageType>;
  FDKCPUType::Pointer feldkamp = FDKCPUType::New();
  feldkamp->SetInput(0, tomographySource->GetOutput());
  feldkamp->SetInput(1, ddf->GetOutput());

  std::cout << "\n\n****** Case 1: positive offset in geometry ******" << std::endl;
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  slp->SetGeometry(geometry);
  ddf->SetGeometry(geometry);
  feldkamp->SetGeometry(geometry);
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages, 120., 0.);
  slp->Update();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkamp->Update());
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.061, 24, 2.0);

  std::cout << "\n\n****** Case 2: negative offset in geometry ******" << std::endl;
  geometry = GeometryType::New();
  slp->SetGeometry(geometry);
  ddf->SetGeometry(geometry);
  feldkamp->SetGeometry(geometry);
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages, -120., 0.);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkamp->Update());
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.061, 24, 2.0);

  std::cout << "\n\n****** Case 3: no displacement ******" << std::endl;
  geometry = GeometryType::New();
  slp->SetGeometry(geometry);
  ddf->SetGeometry(geometry);
  feldkamp->SetGeometry(geometry);
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);
  projectionsSource->SetOrigin(origin);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkamp->Update());
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.061, 24, 2.0);

  std::cout << "\n\n****** Case 4: negative offset in origin ******" << std::endl;
  geometry = GeometryType::New();
  slp->SetGeometry(geometry);
  ddf->SetGeometry(geometry);
  feldkamp->SetGeometry(geometry);
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);
  projectionsSource->SetOrigin(itk::MakePoint(-400., -254., -254.));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkamp->Update());
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.061, 24, 2.0);

  std::cout << "\n\n****** Case 5: positive offset in origin ******" << std::endl;
  projectionsSource->SetOrigin(itk::MakePoint(-100., -254., -254.));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkamp->Update());
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.061, 24, 2.0);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
