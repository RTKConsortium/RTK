#include "rtkTest.h"
#include "rtkDrawGeometricPhantomImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkProjectGeometricPhantomImageFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"

/**
 * \file rtkforbildtest.cxx
 *
 * \brief Functional test for Forbild phantom
 *
 * This test reads in a phantom file in the Forbild format, creates projections,
 * reconstructs them and compares the result to the drawing.
 *
 * \author Simon Rit
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 45;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto tomographySource = ConstantImageSourceType::New();
  auto origin = itk::MakePoint(-29., -29., -29.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(6, 6, 6);
  auto spacing = itk::MakeVector(10., 10., 10.);
#else
  auto size = itk::MakeSize(30, 30, 30);
  auto spacing = itk::MakeVector(2., 2., 2.);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  auto projectionsSource = ConstantImageSourceType::New();
  origin = itk::MakePoint(-29., -29., 0.);
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(6, 6, NumberOfProjectionImages);
#else
  size = itk::MakeSize(30, 30, NumberOfProjectionImages);
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  // Rotation matrix
  rtk::ThreeDCircularProjectionGeometry::Matrix3x3Type rotMat;
  rotMat.Fill(0.);
  rotMat[0][0] = 1.;
  rotMat[1][2] = 1.;
  rotMat[2][1] = -1.;

  // Geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto geometry = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 0., noProj * 360. / NumberOfProjectionImages);

  std::string configFileName = std::string(RTK_DATA_ROOT) + std::string("/Input/Forbild/thorax.txt");

  // Shepp Logan projections filter
  std::cout << "\n\n****** Projecting ******" << std::endl;
  using ProjectGPType = rtk::ProjectGeometricPhantomImageFilter<OutputImageType, OutputImageType>;
  auto pgp = ProjectGPType::New();
  pgp->SetInput(projectionsSource->GetOutput());
  pgp->SetGeometry(geometry);
  pgp->SetPhantomScale(1.2);
  pgp->SetConfigFile(configFileName);
  pgp->SetRotationMatrix(rotMat);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(pgp->Update());

  // Create a reference object (in this case a 3D phantom reference).
  std::cout << "\n\n****** Drawing ******" << std::endl;
  using DrawGPType = rtk::DrawGeometricPhantomImageFilter<OutputImageType, OutputImageType>;
  auto dgp = DrawGPType::New();
  dgp->SetInput(tomographySource->GetOutput());
  dgp->SetPhantomScale(1.2);
  dgp->SetConfigFile(configFileName);
  dgp->SetRotationMatrix(rotMat);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dgp->Update())

  // FDK reconstruction filtering
  std::cout << "\n\n****** Reconstructing ******" << std::endl;
  using FDKType = rtk::FDKConeBeamReconstructionFilter<OutputImageType>;
  auto feldkamp = FDKType::New();
  feldkamp->SetInput(0, tomographySource->GetOutput());
  feldkamp->SetInput(1, pgp->GetOutput());
  feldkamp->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkamp->Update());

  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dgp->GetOutput(), 0.065, 24, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
