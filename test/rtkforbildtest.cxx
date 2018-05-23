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

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfProjectionImages = 3;
#else
  const unsigned int NumberOfProjectionImages = 45;
#endif

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  origin[0] = -29.;
  origin[1] = -29.;
  origin[2] = -29.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 6;
  size[1] = 6;
  size[2] = 6;
  spacing[0] = 10.;
  spacing[1] = 10.;
  spacing[2] = 10.;
#else
  size[0] = 30;
  size[1] = 30;
  size[2] = 30;
  spacing[0] = 2.;
  spacing[1] = 2.;
  spacing[2] = 2.;
#endif
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -29.;
  origin[1] = -29.;
  origin[2] = 0.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 6;
  size[1] = 6;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 10.;
  spacing[1] = 10.;
  spacing[2] = 10.;
#else
  size[0] = 30;
  size[1] = 30;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 2.;
  spacing[1] = 2.;
  spacing[2] = 2.;
#endif
  projectionsSource->SetOrigin( origin );
  projectionsSource->SetSpacing( spacing );
  projectionsSource->SetSize( size );
  projectionsSource->SetConstant( 0. );

  // Rotation matrix
  rtk::ThreeDCircularProjectionGeometry::Matrix3x3Type rotMat;
  rotMat.Fill(0.);
  rotMat[0][0] = 1.;
  rotMat[1][2] = 1.;
  rotMat[2][1] = -1.;

  // Geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 0., noProj*360./NumberOfProjectionImages);

  std::string configFileName = std::string(RTK_DATA_ROOT) +
                               std::string("/Input/Forbild/thorax.txt");

  // Shepp Logan projections filter
  std::cout << "\n\n****** Projecting ******" << std::endl;
  typedef rtk::ProjectGeometricPhantomImageFilter<OutputImageType, OutputImageType> ProjectGPType;
  ProjectGPType::Pointer pgp = ProjectGPType::New();
  pgp->SetInput( projectionsSource->GetOutput() );
  pgp->SetGeometry(geometry);
  pgp->SetPhantomScale(1.2);
  pgp->SetConfigFile(configFileName);
  pgp->SetIsForbildConfigFile(true);
  pgp->SetRotationMatrix(rotMat);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( pgp->Update() );

  // Create a reference object (in this case a 3D phantom reference).
  std::cout << "\n\n****** Drawing ******" << std::endl;
  typedef rtk::DrawGeometricPhantomImageFilter<OutputImageType, OutputImageType> DrawGPType;
  DrawGPType::Pointer dgp = DrawGPType::New();
  dgp->SetInput( tomographySource->GetOutput() );
  dgp->SetPhantomScale(1.2);
  dgp->SetConfigFile(configFileName);
  dgp->SetIsForbildConfigFile(true);
  dgp->SetRotationMatrix(rotMat);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dgp->Update() )

  // FDK reconstruction filtering
  std::cout << "\n\n****** Reconstructing ******" << std::endl;
  typedef rtk::FDKConeBeamReconstructionFilter< OutputImageType > FDKType;
  FDKType::Pointer feldkamp = FDKType::New();
  feldkamp->SetInput( 0, tomographySource->GetOutput() );
  feldkamp->SetInput( 1, pgp->GetOutput() );
  feldkamp->SetGeometry( geometry );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );

  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dgp->GetOutput(), 0.065, 24, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
