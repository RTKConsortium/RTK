#include <itkImageRegionConstIterator.h>
#include <itkStreamingImageFilter.h>
#include <itkImageRegionSplitterDirection.h>

#include "rtkTest.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"

#ifdef USE_CUDA
#  include "rtkCudaFDKConeBeamReconstructionFilter.h"
#else
#  include "rtkFDKConeBeamReconstructionFilter.h"
#endif

/**
 * \file rtkfdktest.cxx
 *
 * \brief Functional test for classes performing FDK reconstructions
 *
 * This test generates the projections of a simulated Shepp-Logan phantom.
 * A CT image is reconstructed from each set of generated projection images
 * using the FDK algorithm and the reconstructed CT image is compared to the
 * expected results which is analytically computed.
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
  auto origin = itk::MakePoint(-127., -127., -127.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(32, 32, 32);
  auto spacing = itk::MakeVector(8., 8., 8.);
#else
  auto size = itk::MakeSize(128, 128, 128);
  auto spacing = itk::MakeVector(2., 2., 2.);
#endif
  auto tomographySource = ConstantImageSourceType::New();
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  origin = itk::MakePoint(-254., -254., -254.);
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(32, 32, NumberOfProjectionImages);
  spacing = itk::MakeVector(32., 32., 32.);
#else
  size = itk::MakeSize(128, 128, NumberOfProjectionImages);
  spacing = itk::MakeVector(4., 4., 4.);
#endif
  auto projectionsSource = ConstantImageSourceType::New();
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  std::cout << "\n\n****** Case 1: No streaming ******" << std::endl;

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
  using FDKType = rtk::CudaFDKConeBeamReconstructionFilter;
#else
  using FDKType = rtk::FDKConeBeamReconstructionFilter<OutputImageType>;
#endif
  auto feldkamp = FDKType::New();
  feldkamp->SetInput(0, tomographySource->GetOutput());
  feldkamp->SetInput(1, slp->GetOutput());
  feldkamp->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkamp->Update());


  // FOV
  using FOVFilterType = rtk::FieldOfViewImageFilter<OutputImageType, OutputImageType>;
  auto fov = FOVFilterType::New();
  fov->SetInput(0, feldkamp->GetOutput());
  fov->SetProjectionsStack(slp->GetOutput());
  fov->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(fov->Update());

  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput(), 0.03, 26, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: perpendicular direction ******" << std::endl;

  ConstantImageSourceType::OutputImageType::DirectionType direction;
  direction[0][0] = 0;
  direction[0][1] = 1;
  direction[0][2] = 0;
  direction[1][0] = -1;
  direction[1][1] = 0;
  direction[1][2] = 0;
  direction[2][0] = 0;
  direction[2][1] = 0;
  direction[2][2] = 1;
  tomographySource->SetDirection(direction);
  tomographySource->SetOrigin(itk::MakePoint(-127., 127., -127.));
  fov->GetOutput()->ResetPipeline();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(fov->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update())

  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput(), 0.03, 26, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  std::cout << "\n\n****** Case 3: 45 degree tilt direction ******" << std::endl;

  direction[0][0] = 0.70710678118;
  direction[0][1] = -0.70710678118;
  direction[0][2] = 0.70710678118;
  direction[1][0] = 0.70710678118;
  direction[1][1] = 0;
  direction[1][2] = 0;
  direction[2][0] = 0;
  direction[2][1] = 0;
  direction[2][2] = 1;
  tomographySource->SetDirection(direction);
  tomographySource->SetOrigin(itk::MakePoint(-127., -127., -127.));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update())
  TRY_AND_EXIT_ON_ITK_EXCEPTION(fov->Update());

  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput(), 0.03, 26, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  std::cout << "\n\n****** Case 4: streaming ******" << std::endl;

  // Make sure that the data will be recomputed by releasing them
  fov->GetOutput()->ReleaseData();

  using StreamingType = itk::StreamingImageFilter<OutputImageType, OutputImageType>;
  auto streamer = StreamingType::New();
  streamer->SetInput(0, fov->GetOutput());
  streamer->SetNumberOfStreamDivisions(8);
  auto splitter = itk::ImageRegionSplitterDirection::New();
  splitter->SetDirection(2); // Prevent splitting along z axis. As a result, splitting will be performed along y axis
  streamer->SetRegionSplitter(splitter);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(streamer->Update());

  CheckImageQuality<OutputImageType>(streamer->GetOutput(), dsl->GetOutput(), 0.03, 26, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  std::cout << "\n\n****** Case 5: small ROI ******" << std::endl;
  tomographySource->SetOrigin(itk::MakePoint(-5., -13., -20.));
  tomographySource->SetSize(itk::MakeSize(64, 64, 64));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(fov->UpdateLargestPossibleRegion());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->UpdateLargestPossibleRegion())
  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput(), 0.03, 26, 2.0);
  std::cout << "Test PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
