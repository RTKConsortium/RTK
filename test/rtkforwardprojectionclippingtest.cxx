#include "rtkTest.h"
#include "rtkConstantImageSource.h"
#include "rtkThreeDCircularProjectionGeometry.h"

#ifdef USE_CUDA
#  include "rtkCudaForwardProjectionImageFilter.h"
#else
#  include "rtkJosephForwardProjectionImageFilter.h"
#endif

#include <cstdlib>
#include <iostream>

int
rtkforwardprojectionclippingtest(int, char *[])
{
  constexpr unsigned int Dimension = 3;
  using PixelType = float;

#ifdef USE_CUDA
  using ImageType = itk::CudaImage<PixelType, Dimension>;
  using ForwardProjectorType = rtk::CudaForwardProjectionImageFilter<ImageType, ImageType>;
#else
  using ImageType = itk::Image<PixelType, Dimension>;
  using ForwardProjectorType = rtk::JosephForwardProjectionImageFilter<ImageType, ImageType>;
#endif

  using ConstantImageSourceType = rtk::ConstantImageSource<ImageType>;

  // Constant 10 mm^3 volume centered at the origin of the fixed coordinate system (FCS)
  auto vol = ConstantImageSourceType::New();
  vol->SetOrigin(itk::MakePoint(-5.0, -5.0, -5.0));
  vol->SetSpacing(itk::MakeVector(1.0, 1.0, 1.0));
  vol->SetSize(itk::MakeSize(10, 10, 10));
  vol->SetConstant(1.0f);

  // Single-pixel flat panel
  auto proj = ConstantImageSourceType::New();
  proj->SetOrigin(itk::MakePoint(0.0, 0.0, 0.0));
  proj->SetSpacing(itk::MakeVector(1.0, 1.0, 1.0));
  proj->SetSize(itk::MakeSize(1, 1, 1));
  proj->SetConstant(0.0f);

  // Single projection, axis-aligned geometry
  // X-ray source is at (0, 0, 60) in FCS
  // Detector pixel center is at (0, 0, -40) in FCS
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto geometry = GeometryType::New();
  geometry->AddProjection(60.0, 100.0, 0.0);

  auto projector = ForwardProjectorType::New();
  projector->SetInput(0, proj->GetOutput());
  projector->SetInput(1, vol->GetOutput());
  projector->SetGeometry(geometry);

  projector->Update();

  constexpr auto absTol{ 1e-5f };

  const auto attenuationBeforeDetector = projector->GetOutput()->GetPixel({});
  if (std::abs(attenuationBeforeDetector - 9.0f) > absTol)
  {
    std::cerr << "Test failed, ray integral when volume is completely inside source-detector segment is "
              << attenuationBeforeDetector << " instead of 9" << std::endl;
    return EXIT_FAILURE;
  }

  // Translate the volume so it is now centered on (0, 0, -40), i.e. the detector pixel
  vol->SetOrigin(itk::MakePoint(-5.0, -5.0, -45.0));
  projector->Update();

  const auto attenuationAtDetector = projector->GetOutput()->GetPixel({});
  if (std::abs(attenuationAtDetector - 4.0f) > absTol)
  {
    std::cerr << "Test failed, ray integral when half of volume is inside source-detector segment is "
              << attenuationAtDetector << " instead of 4" << std::endl;
    return EXIT_FAILURE;
  }

  // Translate again the volume: it is now completely outside the source-detector segment
  vol->SetOrigin(itk::MakePoint(-5.0, -5.0, -85.0));
  projector->Update();

  const auto attenuationBeyondDetector = projector->GetOutput()->GetPixel({});
  if (std::abs(attenuationBeyondDetector - 0.0f) > absTol)
  {
    std::cerr << "Test failed, ray integral when volume is outside source-detector segment is "
              << attenuationBeyondDetector << " instead of 0" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
