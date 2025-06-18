#include "rtkTest.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"

#include <itkStreamingImageFilter.h>
#include <itkImageRegionSplitterDirection.h>

#ifdef USE_CUDA
#  include "rtkCudaForwardProjectionImageFilter.h"
#else
#  include "rtkJosephForwardProjectionImageFilter.h"
#endif

/**
 * \file rtkforwardprojectiontest.cxx
 *
 * \brief Functional test for forward projection
 *
 * The test projects a volume filled with ones. The forward projector should
 * then return the intersection of the ray with the box and it is compared
 * with the analytical intersection of a box with a ray.
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

  using VectorType = itk::Vector<double, 3>;
#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 45;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  // Create Joseph Forward Projector volume input.
  const ConstantImageSourceType::Pointer volInput = ConstantImageSourceType::New();
  auto                                   origin = itk::MakePoint(-126., -126., -126.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(2, 2, 2);
  auto spacing = itk::MakeVector(252., 252., 252.);
#else
  auto size = itk::MakeSize(64, 64, 64);
  auto spacing = itk::MakeVector(4., 4., 4.);
#endif
  volInput->SetOrigin(origin);
  volInput->SetSpacing(spacing);
  volInput->SetSize(size);
  volInput->SetConstant(1.);
  volInput->UpdateOutputInformation();

  // Initialization Volume, it is used in the Joseph Forward Projector and in the
  // Ray Box Intersection Filter in order to initialize the stack of projections.
  const ConstantImageSourceType::Pointer projInput = ConstantImageSourceType::New();
  size[2] = NumberOfProjectionImages;
  projInput->SetOrigin(origin);
  projInput->SetSpacing(spacing);
  projInput->SetSize(size);
  projInput->SetConstant(0.);
  projInput->Update();

  // Joseph Forward Projection filter
#ifdef USE_CUDA
  using JFPType = rtk::CudaForwardProjectionImageFilter<OutputImageType, OutputImageType>;
#else
  using JFPType = rtk::JosephForwardProjectionImageFilter<OutputImageType, OutputImageType>;
#endif
  JFPType::Pointer jfp = JFPType::New();
  jfp->InPlaceOff();
  jfp->SetInput(projInput->GetOutput());
  jfp->SetInput(1, volInput->GetOutput());

  // Ray Box Intersection filter (reference)
  using RBIType = rtk::RayBoxIntersectionImageFilter<OutputImageType, OutputImageType>;
#ifdef USE_CUDA
  jfp->SetStepSize(10);
#endif
  RBIType::Pointer rbi = RBIType::New();
  rbi->InPlaceOff();
  rbi->SetInput(projInput->GetOutput());
  rbi->SetBoxMin(itk::MakeVector(-126.0, -126.0, -126.0));
  rbi->SetBoxMax(itk::MakeVector(126.0, 126.0, 47.6));

  // Streaming filter to test for unusual regions
  using StreamingFilterType = itk::StreamingImageFilter<OutputImageType, OutputImageType>;
  StreamingFilterType::Pointer stream = StreamingFilterType::New();
  stream->SetInput(jfp->GetOutput());

  stream->SetNumberOfStreamDivisions(9);
  itk::ImageRegionSplitterDirection::Pointer splitter = itk::ImageRegionSplitterDirection::New();
  splitter->SetDirection(2); // Splitting along direction 1, NOT 2
  stream->SetRegionSplitter(splitter);

  std::cout << "\n\n****** Case 1: inner ray source ******" << std::endl;
  // The circle is divided in 4 quarters
  for (int q = 0; q < 4; q++)
  {
    // Geometry
    using GeometryType = rtk::ThreeDCircularProjectionGeometry;
    GeometryType::Pointer geometry = GeometryType::New();
    for (unsigned int i = 0; i < NumberOfProjectionImages; i++)
    {
      const double angle = -45. + i * 2.;
      geometry->AddProjection(47.6 / std::cos(angle * itk::Math::pi / 180.), 1000., q * 90 + angle);
    }

    if (q == 0)
    {
      rbi->SetGeometry(geometry);
      rbi->Update();
    }

    jfp->SetGeometry(geometry);
    stream->Update();

    CheckImageQuality<OutputImageType>(stream->GetOutput(), rbi->GetOutput(), 1.28, 44.0, 255.0);
    std::cout << "\n\nTest of quarter #" << q << " PASSED! " << std::endl;
  }

#ifdef USE_CUDA
  jfp->SetStepSize(1);
#endif

  std::cout << "\n\n****** Case 2: outer ray source ******" << std::endl;
  rbi->SetBoxMax(itk::MakeVector(126.0, 126.0, 126.0));

  // Geometry
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  for (unsigned int i = 0; i < NumberOfProjectionImages; i++)
    geometry->AddProjection(500., 1000., i * 8.);

  rbi->SetGeometry(geometry);
  rbi->Update();

  jfp->SetGeometry(geometry);
  stream->Update();

  CheckImageQuality<OutputImageType>(stream->GetOutput(), rbi->GetOutput(), 1.28, 44.0, 255.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 3: Shepp-Logan, outer ray source ******" << std::endl;

  // Create Shepp Logan reference projections
  using SLPType = rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType>;
  SLPType::Pointer slp = SLPType::New();
  slp->InPlaceOff();
  slp->SetInput(projInput->GetOutput());
  slp->SetGeometry(geometry);
  slp->Update();

  // Create a Shepp Logan reference volume (finer resolution)
  origin.Fill(-127);
  size.Fill(128);
  spacing.Fill(2.);
  volInput->SetOrigin(origin);
  volInput->SetSpacing(spacing);
  volInput->SetSize(size);
  volInput->SetConstant(0.);

  using DSLType = rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType>;
  DSLType::Pointer dsl = DSLType::New();
  dsl->InPlaceOff();
  dsl->SetInput(volInput->GetOutput());
  dsl->Update();

  // Forward projection
  jfp->SetInput(1, dsl->GetOutput());
  stream->Update();

  CheckImageQuality<OutputImageType>(stream->GetOutput(), slp->GetOutput(), 1.28, 44, 255.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 4: Shepp-Logan, outer ray source, cylindrical detector ******" << std::endl;
  geometry->SetRadiusCylindricalDetector(600);

  slp->SetGeometry(geometry);
  slp->Update();

  jfp->SetGeometry(geometry);
  stream->Update();

  CheckImageQuality<OutputImageType>(stream->GetOutput(), slp->GetOutput(), 1.28, 44, 255.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 5: Shepp-Logan, inner ray source ******" << std::endl;
  geometry = GeometryType::New();
  for (unsigned int i = 0; i < NumberOfProjectionImages; i++)
    geometry->AddProjection(120., 1000., i * 8.);

  slp->SetGeometry(geometry);
  slp->Update();

  jfp->SetGeometry(geometry);
  stream->Update();

  CheckImageQuality<OutputImageType>(stream->GetOutput(), slp->GetOutput(), 1.28, 44, 255.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
