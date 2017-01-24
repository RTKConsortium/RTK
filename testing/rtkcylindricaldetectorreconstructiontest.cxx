#include "rtkTest.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkConjugateGradientConeBeamReconstructionFilter.h"

#ifdef USE_CUDA
  #include "itkCudaImage.h"
#endif

/**
 * \file rtkcylindricaldetectorreconstructiontest.cxx
 *
 * \brief Functional test for conjugate gradient reconstruction with cylindrical-detector
 *
 * This test generates the projections of an ellipsoid onto a cylindrical detector,
 * and reconstructs the CT image using the conjugate gradient algorithm
 * with different backprojectors (Joseph, CudaRayCast). The generated results are compared to the
 * expected results (analytical calculation).
 *
 * \author Cyril Mory
 */

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
#endif

#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfProjectionImages = 3;
#else
  const unsigned int NumberOfProjectionImages = 180;
#endif

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;
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
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -255.;
  origin[1] = -255.;
  origin[2] = -255.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 504.;
  spacing[1] = 504.;
  spacing[2] = 504.;
#else
  size[0] = 64;
  size[1] = 64;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
#endif
  projectionsSource->SetOrigin( origin );
  projectionsSource->SetSpacing( spacing );
  projectionsSource->SetSize( size );
  projectionsSource->SetConstant( 0. );

  // Geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  geometry->SetRadiusCylindricalDetector(200);
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages);

  // Create ellipsoid PROJECTIONS
  typedef rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType> REIType;
  REIType::Pointer rei;

  rei = REIType::New();
  REIType::VectorType semiprincipalaxis, center;
  semiprincipalaxis.Fill(90.);
  center.Fill(0.);
  rei->SetAngle(0.);
  rei->SetDensity(1.);
  rei->SetCenter(center);
  rei->SetAxis(semiprincipalaxis);

  rei->SetInput( projectionsSource->GetOutput() );
  rei->SetGeometry( geometry );

  //Update
  TRY_AND_EXIT_ON_ITK_EXCEPTION( rei->Update() );

  // Create REFERENCE object (3D ellipsoid).
  typedef rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType> DEType;
  DEType::Pointer dsl = DEType::New();
  dsl->SetInput( tomographySource->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() )

  // Define weights (for weighted least squares)
  ConstantImageSourceType::Pointer uniformWeightsSource = ConstantImageSourceType::New();
  uniformWeightsSource->SetInformationFromImage(projectionsSource->GetOutput());
  uniformWeightsSource->SetConstant(1.0);

  // ConjugateGradient reconstruction filtering
  typedef rtk::ConjugateGradientConeBeamReconstructionFilter< OutputImageType > ConjugateGradientType;
  ConjugateGradientType::Pointer conjugategradient = ConjugateGradientType::New();
  conjugategradient->SetInput( tomographySource->GetOutput() );
  conjugategradient->SetInput(1, rei->GetOutput());
  conjugategradient->SetInput(2, uniformWeightsSource->GetOutput());
  conjugategradient->SetGeometry( geometry );
  conjugategradient->SetNumberOfIterations( 5 );
  conjugategradient->SetRegularized(false);
  conjugategradient->SetDisableDisplacedDetectorFilter(true);

  std::cout << "\n\n****** Case 1: Joseph forward and back projectors ******" << std::endl;

  conjugategradient->SetForwardProjectionFilter(0);
  conjugategradient->SetBackProjectionFilter(1);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( conjugategradient->Update() );

  CheckImageQuality<OutputImageType>(conjugategradient->GetOutput(), dsl->GetOutput(), 0.08, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 2: CUDA ray cast forward and back projectors ******" << std::endl;

  conjugategradient->SetForwardProjectionFilter(1);
  conjugategradient->SetBackProjectionFilter(4);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( conjugategradient->Update() );

  CheckImageQuality<OutputImageType>(conjugategradient->GetOutput(), dsl->GetOutput(), 0.08, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  std::cout << "\n\n****** Case 3: Joseph forward projector, voxel-based back projector ******" << std::endl;

  // Only works with a cylindrical detector centered on the source, therefore a new geometry has to be simulated
  geometry = GeometryType::New();
  geometry->SetRadiusCylindricalDetector(1200);
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages);

  conjugategradient->SetGeometry(geometry);
  conjugategradient->SetForwardProjectionFilter(0);
  conjugategradient->SetBackProjectionFilter(0);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( conjugategradient->Update() );

  CheckImageQuality<OutputImageType>(conjugategradient->GetOutput(), dsl->GetOutput(), 0.08, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 4: CUDA ray cast forward projection, CUDA voxel based back projection ******" << std::endl;

  conjugategradient->SetForwardProjectionFilter(1);
  conjugategradient->SetBackProjectionFilter(2);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( conjugategradient->Update() );

  CheckImageQuality<OutputImageType>(conjugategradient->GetOutput(), dsl->GetOutput(), 0.08, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif


  return EXIT_SUCCESS;
}
