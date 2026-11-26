#include "rtkTest.h"
#include "rtkConstantImageSource.h"
#include "rtkDrawBoxImageFilter.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkDrawGeometricPhantomImageFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkMacro.h"
#include "rtkQuadricShape.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

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
rtkdrawgeometricphantomtest(int argc, char * argv[])
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  SheppLoganConfig.txt GeometryConfig.txt" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 3;
  using OutputImageType = itk::Image<float, Dimension>;

  // Constant image sources
  auto tomographySource = rtk::ConstantImageSource<OutputImageType>::New();
  auto origin = itk::MakePoint(-127., -127., -127.);
#if FAST_TESTS_NO_CHECKS
  auto spacing = itk::MakeVector(254., 254., 254.);
  auto size = itk::MakeSize(2, 2, 2);
#else
  auto spacing = itk::MakeVector(2., 2., 2.);
  auto size = itk::MakeSize(128, 128, 128);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  //////////////////////////////////
  // Part 1: Shepp Logan
  //////////////////////////////////

  // Shepp Logan reference filter
  auto dsl = rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType>::New();
  dsl->SetInput(tomographySource->GetOutput());
  dsl->SetPhantomScale(128.);
  dsl->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update());

  // Shepp Logan reference filter from Configuration File
  auto dgp = rtk::DrawGeometricPhantomImageFilter<OutputImageType, OutputImageType>::New();
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

  // Draw cylinder, i.e., an ellipsoid with an infinite axis
  auto dcl = rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType>::New();
  dcl->SetInput(tomographySource->GetOutput());
  dcl->SetAxis(itk::MakeVector(100., 0., 100.));
  dcl->SetCenter(itk::MakePoint(2., 2., 2.));
  dcl->SetAngle(0.);
  dcl->SetDensity(2.);
  dcl->InPlaceOff();

  // Draw cone, i.e., an ellipsoid quadric without constant term
  auto cone = rtk::QuadricShape::New();
  cone->SetEllipsoid(itk::MakePoint(2., 2., 2.), itk::MakeVector(25., -50., 25.), 0.);
  cone->SetJ(0.);
  cone->SetDensity(-0.54);
  auto dco = rtk::DrawConvexImageFilter<OutputImageType, OutputImageType>::New();
  dco->SetInput(tomographySource->GetOutput());
  dco->SetConvexShape(cone);

  // Add Image Filter used to concatenate the different figures obtained on each iteration
  auto addFilter = itk::AddImageFilter<OutputImageType, OutputImageType, OutputImageType>::New();

  addFilter->SetInput1(dcl->GetOutput());
  addFilter->SetInput2(dco->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(addFilter->Update());

  CheckImageQuality<OutputImageType>(dgp->GetOutput(), addFilter->GetOutput(), 0.0005, 90, 255.0);
  std::cout << "Test PASSED! " << std::endl;

  //////////////////////////////////
  // Part 3: box covering the image, then clipped
  //////////////////////////////////

  TRY_AND_EXIT_ON_ITK_EXCEPTION(tomographySource->UpdateOutputInformation());
  auto dbo = rtk::DrawBoxImageFilter<OutputImageType, OutputImageType>::New();
  dbo->SetInput(tomographySource->GetOutput());
  dbo->SetBoxFromImage(tomographySource->GetOutput());
  dbo->SetDensity(3.);
  dbo->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dbo->Update());

  // Voxels at the two extremities along x, which are on both sides of x=0
  OutputImageType::IndexType firstIndex = tomographySource->GetOutput()->GetLargestPossibleRegion().GetIndex();
  OutputImageType::IndexType lastIndex = firstIndex;
  for (unsigned int i = 0; i < Dimension; i++)
    lastIndex[i] += size[i] - 1;
  if (dbo->GetOutput()->GetPixel(firstIndex) != 3. || dbo->GetOutput()->GetPixel(lastIndex) != 3.)
  {
    std::cerr << "Test FAILED! The box does not cover the image." << std::endl;
    return EXIT_FAILURE;
  }

  // Keep the half space x<0, the filter must be updated
  dbo->AddClipPlane(itk::MakeVector(1., 0., 0.), 0.);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dbo->Update());
  if (dbo->GetOutput()->GetPixel(firstIndex) != 3. || dbo->GetOutput()->GetPixel(lastIndex) != 0.)
  {
    std::cerr << "Test FAILED! The clip plane has not been applied to the box." << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Test PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
