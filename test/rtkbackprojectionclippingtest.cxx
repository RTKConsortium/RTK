#include "rtkTest.h"
#include "rtkConstantImageSource.h"
#include "rtkThreeDCircularProjectionGeometry.h"

#ifdef USE_CUDA
#  include "rtkCudaRayCastBackProjectionImageFilter.h"
#else
#  include "rtkJosephBackProjectionImageFilter.h"
#endif

#include <cstdlib>
#include <iostream>

int
rtkbackprojectionclippingtest(int, char *[])
{
  constexpr unsigned int Dimension = 3;
  using PixelType = float;

#ifdef USE_CUDA
  using ImageType = itk::CudaImage<PixelType, Dimension>;
  using BackProjectorType = rtk::CudaRayCastBackProjectionImageFilter;
#else
  using ImageType = itk::Image<PixelType, Dimension>;
  using BackProjectorType = rtk::JosephBackProjectionImageFilter<ImageType, ImageType>;
#endif

  using ConstantImageSourceType = rtk::ConstantImageSource<ImageType>;

  // Constant 3 mm^3 volume centered at the origin of the fixed coordinate system (FCS)
  auto vol = ConstantImageSourceType::New();
  vol->SetOrigin(itk::MakePoint(-1.0, -1.0, -1.0));
  vol->SetSpacing(itk::MakeVector(1.0, 1.0, 1.0));
  vol->SetSize(itk::MakeSize(3, 3, 3));
  // Index of the volume center
  const auto centerVoxelIndex = itk::MakeIndex(1, 1, 1);

  // Single-pixel flat panel
  auto proj = ConstantImageSourceType::New();
  proj->SetOrigin(itk::MakePoint(0.0, 0.0, 0.0));
  proj->SetSpacing(itk::MakeVector(1.0, 1.0, 1.0));
  proj->SetSize(itk::MakeSize(1, 1, 1));
  constexpr auto pixelValue{ 1.0f };
  proj->SetConstant(pixelValue);

  // Single projection, axis-aligned geometry
  // X-ray source is at (0, 0, 60) in FCS
  // Detector pixel center is at (0, 0, -40) in FCS
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto geometry = GeometryType::New();
  geometry->AddProjection(60.0, 100.0, 0.0);

  auto projector = BackProjectorType::New();
  projector->SetInput(0, vol->GetOutput());
  projector->SetInput(1, proj->GetOutput());
  projector->SetGeometry(geometry);

  projector->Update();

  constexpr auto absTol{ 1e-5f };

  const auto valueBeforeDetector = projector->GetOutput()->GetPixel(centerVoxelIndex);
  if (std::abs(valueBeforeDetector - pixelValue) > absTol)
  {
    std::cerr << "Test failed, back-projected value when volume is completely inside source-detector segment is "
              << valueBeforeDetector << " instead of " << pixelValue << std::endl;
    return EXIT_FAILURE;
  }

  // Translate the volume outside the source-detector segment, while keeping its center along the z-axis
  // (source-detector line)
  vol->SetOrigin(itk::MakePoint(-1.0, -1.0, -81.0));
  projector->Update();

  const auto valueBeyondDetector = projector->GetOutput()->GetPixel(centerVoxelIndex);
  if (std::abs(valueBeyondDetector - 0.0f) > absTol)
  {
    std::cerr << "Test failed, back-projected value when volume is outside source-detector segment is "
              << valueBeyondDetector << " instead of 0" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
