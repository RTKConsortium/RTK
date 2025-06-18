#include <itkImageRegionConstIterator.h>

#include "rtkTest.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkConstantImageSource.h"
#ifdef USE_CUDA
#  include "rtkCudaParkerShortScanImageFilter.h"
#else
#  include "rtkParkerShortScanImageFilter.h"
#endif

/**
 * \file rtkshortscantest.cxx
 *
 * \brief Functional test for FDK reconstruction from short scan
 *
 * This test generates the projections of a simulated Shepp-Logan phantom with
 * a short scan geometry. The corresponding CT image is reconstructed using
 * FDK with Parker weighting. The generated results are compared to the
 * expected results (analytical calculation).
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
  constexpr unsigned int NumberOfProjectionImages = 110;
#endif
  constexpr double ArcSize = 240.;

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

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
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
  GeometryType::Pointer geometry = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * ArcSize / NumberOfProjectionImages);

  // Shepp Logan projections filter
  using SLPType = rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType>;
  SLPType::Pointer slp = SLPType::New();
  slp->SetInput(projectionsSource->GetOutput());
  slp->SetGeometry(geometry);
  slp->SetPhantomScale(116);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(slp->Update());

  // Short scan image filter
#ifdef USE_CUDA
  using PSSFType = rtk::CudaParkerShortScanImageFilter;
#else
  using PSSFType = rtk::ParkerShortScanImageFilter<OutputImageType>;
#endif
  PSSFType::Pointer pssf = PSSFType::New();
  pssf->SetInput(slp->GetOutput());
  pssf->SetGeometry(geometry);
  pssf->InPlaceOff();
  pssf->Update();

  // Create a reference object (in this case a 3D phantom reference).
  using DSLType = rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType>;
  DSLType::Pointer dsl = DSLType::New();
  dsl->SetInput(tomographySource->GetOutput());
  dsl->SetPhantomScale(116);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update())

  // FDK reconstruction filtering
  using FDKCPUType = rtk::FDKConeBeamReconstructionFilter<OutputImageType>;
  FDKCPUType::Pointer feldkamp = FDKCPUType::New();
  feldkamp->SetInput(0, tomographySource->GetOutput());
  feldkamp->SetInput(1, pssf->GetOutput());
  feldkamp->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkamp->Update());

  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.09, 22, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
