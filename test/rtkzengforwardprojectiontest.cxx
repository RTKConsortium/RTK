#include "rtkTest.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkJosephForwardAttenuatedProjectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkZengForwardProjectionImageFilter.h"
#include <itkImageRegionSplitterDirection.h>
#include <itkImageRegionIterator.h>
#include <cmath>
#include "rtkDrawEllipsoidImageFilter.h"

/**
 * \file rtkzengforwardprojectiontest.cxx
 *
 * \brief Functional test for forward projection
 *
 * The test projects a volume filled with ones. The forward projector should
 * then return the intersection of the ray with the box and it is compared
 * with the analytical intersection of a box with a ray.
 *
 * \author Antoine Robert
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
  constexpr unsigned int NumberOfProjectionImages = 40;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  constexpr double att = 0.0154;
  // Create Joseph Forward Projector volume input.
  auto origin = itk::MakePoint(-126., -126., -126.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(2, 2, 2);
  auto spacing = itk::MakeVector(252., 252., 252.);
#else
  auto size = itk::MakeSize(64, 64, 64);
  auto spacing = itk::MakeVector(4., 4., 4.);
#endif

  auto tomographySource = ConstantImageSourceType::New();
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  auto attenuationInput = ConstantImageSourceType::New();
  attenuationInput->SetOrigin(origin);
  attenuationInput->SetSpacing(spacing);
  attenuationInput->SetSize(size);
  attenuationInput->SetConstant(att);


  // Initialization Volume, it is used in the Zeng Forward Projector and in the Joseph Attenuated Projector
  auto projInput = ConstantImageSourceType::New();
  size[2] = NumberOfProjectionImages;
  projInput->SetOrigin(origin);
  projInput->SetSpacing(spacing);
  projInput->SetSize(size);
  projInput->SetConstant(0.);
  projInput->Update();


  auto volInput = rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType>::New();
  auto axis_vol = itk::MakeVector(32., 32., 32.);
  auto center_vol = itk::MakePoint(0., 0., 0.);
  volInput->SetInput(tomographySource->GetOutput());
  volInput->SetCenter(center_vol);
  volInput->SetAxis(axis_vol);
  volInput->SetDensity(1);
  volInput->Update();

  // Zeng Forward Projection filter

  auto jfp = rtk::ZengForwardProjectionImageFilter<OutputImageType, OutputImageType>::New();
  jfp->InPlaceOff();
  jfp->SetInput(projInput->GetOutput());
  jfp->SetInput(1, volInput->GetOutput());
  jfp->SetInput(2, attenuationInput->GetOutput());

  // Joseph Forward Attenuated Projection filter
  auto attjfp = rtk::JosephForwardAttenuatedProjectionImageFilter<OutputImageType, OutputImageType>::New();
  attjfp->InPlaceOff();
  attjfp->SetInput(projInput->GetOutput());
  attjfp->SetInput(1, volInput->GetOutput());
  attjfp->SetInput(2, attenuationInput->GetOutput());

  auto geometry = rtk::ThreeDCircularProjectionGeometry::New();
  for (unsigned int i = 0; i < NumberOfProjectionImages; i++)
  {
    geometry->AddProjection(500, 0, i * 360. / NumberOfProjectionImages, 16., 12.);
  }

  attjfp->SetGeometry(geometry);
  attjfp->Update();

  jfp->SetGeometry(geometry);
  jfp->SetAlpha(0.);
  jfp->SetSigmaZero(0.);
  jfp->Update();

  CheckImageQuality<OutputImageType>(jfp->GetOutput(), attjfp->GetOutput(), 0.1, 44.0, 255.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
