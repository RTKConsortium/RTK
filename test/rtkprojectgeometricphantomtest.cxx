#include "rtkTest.h"
#include "rtkConstantImageSource.h"
#include "rtkForbildPhantomFileReader.h"
#include "rtkMacro.h"
#include "rtkProjectGeometricPhantomImageFilter.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

#include <itkRegularExpressionSeriesFileNames.h>

/**
 * \file rtkprojectgeometricphantomtest.cxx
 *
 * \brief Functional test for the class that projects a geometric phantom
 * specified in a config file.
 *
 * This test generates the projections of a Shepp-Logan phantom which are
 * specified by a configuration file located at the Baseline folder.
 * The generated results are compared to the expected results, which are
 * created through hard-coded geometric parameters.
 *
 * \author Marc Vila
 */

int
rtkprojectgeometricphantomtest(int argc, char * argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  ConfigFile.txt" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 3;
  using OutputImageType = itk::Image<float, Dimension>;
#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 180;
#endif
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;

  // Constant image sources
  auto projectionsSource = rtk::ConstantImageSource<OutputImageType>::New();
  auto origin = itk::MakePoint(-254., -254., -254.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(2, 2, NumberOfProjectionImages);
  auto spacing = itk::MakeVector(508., 508., 508.);
#else
  auto size = itk::MakeSize(128, 128, NumberOfProjectionImages);
  auto spacing = itk::MakeVector(4., 4., 4.);
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  // Geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto geometry = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);

  // Shepp Logan projections filter
  auto slp = rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType>::New();
  slp->SetInput(projectionsSource->GetOutput());
  slp->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(slp->Update());

  // Shepp Logan projections filter from Configuration File
  auto pgp = rtk::ProjectGeometricPhantomImageFilter<OutputImageType, OutputImageType>::New();
  pgp->SetInput(projectionsSource->GetOutput());
  pgp->SetGeometry(geometry);
  pgp->SetConfigFile(argv[1]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(pgp->Update());

  CheckImageQuality<OutputImageType>(slp->GetOutput(), pgp->GetOutput(), 0.00055, 88, 255.0);
  std::cout << "Test PASSED! " << std::endl;

  // Phantom set as an input, with a source not shared with filters running in place
  auto reader = rtk::ForbildPhantomFileReader::New();
  reader->SetFilename(argv[1]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->GenerateOutputInformation());
  rtk::GeometricPhantom::Pointer phantom = reader->GetModifiableGeometricPhantom();
  auto                           pgpInputSource = rtk::ConstantImageSource<OutputImageType>::New();
  pgpInputSource->SetOrigin(origin);
  pgpInputSource->SetSpacing(spacing);
  pgpInputSource->SetSize(size);
  pgpInputSource->SetConstant(0.);
  auto pgpInput = rtk::ProjectGeometricPhantomImageFilter<OutputImageType, OutputImageType>::New();
  pgpInput->SetInput(pgpInputSource->GetOutput());
  pgpInput->SetGeometry(geometry);
  pgpInput->SetGeometricPhantom(phantom);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(pgpInput->Update());
  CheckImageQuality<OutputImageType>(slp->GetOutput(), pgpInput->GetOutput(), 0.00055, 88, 255.0);
  std::cout << "Test PASSED! " << std::endl;

  // Translating the phantom must update the projections, same as OriginOffset
  auto translation = itk::MakeVector(10., 0., 0.);
  phantom->Translate(translation);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(pgpInput->Update());
  pgp->SetOriginOffset(translation);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(pgp->Update());
  CheckImageQuality<OutputImageType>(pgp->GetOutput(), pgpInput->GetOutput(), 0.00055, 88, 255.0);
  std::cout << "Test PASSED! " << std::endl;

  // Modifying a convex shape of the phantom must also update the projections
  const itk::ModifiedTimeType updateTime = pgpInput->GetOutput()->GetUpdateMTime();
  rtk::ConvexShape *          shape = phantom->GetConvexShapes()[0];
  shape->SetDensity(shape->GetDensity() + 1.);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(pgpInput->Update());
  if (pgpInput->GetOutput()->GetUpdateMTime() == updateTime)
  {
    std::cerr << "Test FAILED! The projections have not been updated after modifying a convex shape." << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Test PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
