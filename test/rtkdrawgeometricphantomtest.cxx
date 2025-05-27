#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkConstantImageSource.h"
#include "rtkDrawGeometricPhantomImageFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkDrawCylinderImageFilter.h"
#include "rtkDrawConeImageFilter.h"

#include <itkRegularExpressionSeriesFileNames.h>

using GeometryType = rtk::ThreeDCircularProjectionGeometry;

/**
 * \file rtkdrawgeometricphantomtest.cxx
 *
 * \brief Functional test for the class that creates a geometric phantom
 * specified in a config file.
 *
 * This test generates several phantoms with different geometrical shapes
 * (Cone, Cylinder, Shepp-Logan...) specified by configuration files.
 * The generated results are compared to the expected results, which are
 * created through hard-coded geometric parameters.
 *
 * \author Marc Vila
 */

int
main(int argc, char * argv[])
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  SheppLoganConfig.txt GeometryConfig.txt" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::PointType   origin;
  ConstantImageSourceType::SizeType    size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource = ConstantImageSourceType::New();
  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = 2;
  spacing[0] = 254.;
  spacing[1] = 254.;
  spacing[2] = 254.;
#else
  size[0] = 128;
  size[1] = 128;
  size[2] = 128;
  spacing[0] = 2.;
  spacing[1] = 2.;
  spacing[2] = 2.;
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  //////////////////////////////////
  // Part 1: Shepp Logan
  //////////////////////////////////

  // Shepp Logan reference filter
  using DSLType = rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType>;
  DSLType::Pointer dsl = DSLType::New();
  dsl->SetInput(tomographySource->GetOutput());
  dsl->SetPhantomScale(128.);
  dsl->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update());

  // Shepp Logan reference filter from Configuration File
  using DGPType = rtk::DrawGeometricPhantomImageFilter<OutputImageType, OutputImageType>;
  DGPType::Pointer dgp = DGPType::New();
  dgp->SetInput(tomographySource->GetOutput());
  dgp->InPlaceOff();
  dgp->SetConfigFile(argv[1]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dgp->Update());

  CheckImageQuality<OutputImageType>(dsl->GetOutput(), dgp->GetOutput(), 0.0005, 90, 255.0);
  std::cout << "Test PASSED! " << std::endl;

  //////////////////////////////////
  // Part 2: other geometries than ellipsoid
  //////////////////////////////////

  // New Geometries from Configuration File
  dgp->SetInput(tomographySource->GetOutput());
  dgp->SetConfigFile(argv[2]);
  dgp->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dgp->Update());

  //    // Create Reference
  //    std::vector< double > axis;
  //    axis.push_back(100.);
  //    axis.push_back(0.);
  //    axis.push_back(100.);

  //    std::vector< double > center;
  //    center.push_back(2.);
  //    center.push_back(2.);
  //    center.push_back(2.);

  // Draw CYLINDER
  using DCType = rtk::DrawCylinderImageFilter<OutputImageType, OutputImageType>;
  DCType::Pointer dcl = DCType::New();

  DCType::VectorType axis, center;
  axis[0] = 100.;
  axis[1] = 0.;
  axis[2] = 100.;
  center[0] = 2.;
  center[1] = 2.;
  center[2] = 2.;

  dcl->SetInput(tomographySource->GetOutput());
  dcl->SetAxis(axis);
  dcl->SetCenter(center);
  dcl->SetAngle(0.);
  dcl->SetDensity(2.);
  dcl->InPlaceOff();

  // Draw CONE
  // axis.clear();
  axis[0] = 25.;
  axis[1] = -50.;
  axis[2] = 25.;

  using DCOType = rtk::DrawConeImageFilter<OutputImageType, OutputImageType>;
  DCOType::Pointer dco = DCOType::New();
  dco->SetInput(tomographySource->GetOutput());
  dco->SetAxis(axis);
  dco->SetCenter(center);
  dco->SetAngle(0.);
  dco->SetDensity(-0.54);

  // Add Image Filter used to concatenate the different figures obtained on each iteration
  using AddImageFilterType = itk::AddImageFilter<OutputImageType, OutputImageType, OutputImageType>;
  AddImageFilterType::Pointer addFilter = AddImageFilterType::New();

  addFilter->SetInput1(dcl->GetOutput());
  addFilter->SetInput2(dco->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(addFilter->Update());

  CheckImageQuality<OutputImageType>(dgp->GetOutput(), addFilter->GetOutput(), 0.0005, 90, 255.0);
  std::cout << "Test PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
