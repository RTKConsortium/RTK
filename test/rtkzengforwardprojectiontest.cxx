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

  using VectorType = itk::Vector<double, 3>;
#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 40;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::PointType   origin;
  ConstantImageSourceType::SizeType    size;
  ConstantImageSourceType::SpacingType spacing;
  constexpr double                     att = 0.0154;
  // Create Joseph Forward Projector volume input.
  origin[0] = -126;
  origin[1] = -126;
  origin[2] = -126;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = 2;
  spacing[0] = 252.;
  spacing[1] = 252.;
  spacing[2] = 252.;
#else
  size[0] = 64;
  size[1] = 64;
  size[2] = 64;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
#endif

ConstantImageSourceType::Pointer tomographySource = ConstantImageSourceType::New();
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

// Create constant attenuation map
  const ConstantImageSourceType::Pointer attenuationInput = ConstantImageSourceType::New();
  attenuationInput->SetOrigin(origin);
  attenuationInput->SetSpacing(spacing);
  attenuationInput->SetSize(size);
  attenuationInput->SetConstant(att);
  

  // Initialization Volume, it is used in the Zeng Forward Projector and in the Joseph Attenuated Projector
  const ConstantImageSourceType::Pointer projInput = ConstantImageSourceType::New();
  size[2] = NumberOfProjectionImages;
  projInput->SetOrigin(origin);
  projInput->SetSpacing(spacing);
  projInput->SetSize(size);
  projInput->SetConstant(0.);
  projInput->Update();



 using DEIFType = rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType>;
  DEIFType::Pointer    volInput = DEIFType::New();
  DEIFType::VectorType axis_vol, center_vol, center_att, axis_att;
  axis_vol[0] = 32;
  axis_vol[1] = 32;
  axis_vol[2] = 32  ;
  center_vol[0] = 0.;
  center_vol[1] = 0.;
  center_vol[2] = 0.;
  volInput->SetInput(tomographySource->GetOutput());
  volInput->SetCenter(center_vol);
  volInput->SetAxis(axis_vol);
  volInput->SetDensity(1);
  volInput->Update();
  


  // Zeng Forward Projection filter
  using JFPType = rtk::ZengForwardProjectionImageFilter<OutputImageType, OutputImageType>;

  JFPType::Pointer jfp = JFPType::New();
  jfp->InPlaceOff();
  jfp->SetInput(projInput->GetOutput());
  jfp->SetInput(1, volInput->GetOutput());
  jfp->SetInput(2, attenuationInput->GetOutput());

  // Joseph Forward Attenuated Projection filter
  using ATTJFPType = rtk::JosephForwardAttenuatedProjectionImageFilter<OutputImageType, OutputImageType>;
  ATTJFPType::Pointer attjfp = ATTJFPType::New();
  attjfp->InPlaceOff();
  attjfp->SetInput(projInput->GetOutput());
  attjfp->SetInput(1, volInput->GetOutput());
  attjfp->SetInput(2, attenuationInput->GetOutput());


 using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
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
