#include "rtkTest.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkNormalizedJosephBackProjectionImageFilter.h"
#include "rtkADMMTotalVariationConeBeamReconstructionFilter.h"
#include "rtkPhaseGatingImageFilter.h"

#ifdef USE_CUDA
  #include "itkCudaImage.h"
#endif

/**
 * \file rtkadmmtotalvariationtest.cxx
 *
 * \brief Functional test for ADMMTotalVariation reconstruction
 *
 * This test generates the projections of an ellipsoid and reconstructs the CT
 * image using the ADMMTotalVariation algorithm with different backprojectors (Voxel-Based,
 * Joseph). The generated results are compared to the
 * expected results (analytical calculation).
 *
 * \author Cyril Mory
 */

int main(int, char** )
{
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

#ifdef USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
  typedef itk::CudaImage< itk::CovariantVector 
      < OutputPixelType, Dimension >, Dimension >                GradientOutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension >     OutputImageType;
  typedef itk::Image< itk::CovariantVector 
      < OutputPixelType, Dimension >, Dimension >                GradientOutputImageType;
#endif

#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfProjectionImages = 3;
#else
  const unsigned int NumberOfProjectionImages = 90;
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

  // ADMMTotalVariation reconstruction filtering
  typedef rtk::ADMMTotalVariationConeBeamReconstructionFilter
    <OutputImageType, GradientOutputImageType>                ADMMTotalVariationType;
  ADMMTotalVariationType::Pointer admmtotalvariation = ADMMTotalVariationType::New();
  admmtotalvariation->SetInput( tomographySource->GetOutput() );
  admmtotalvariation->SetInput(1, rei->GetOutput());
  admmtotalvariation->SetGeometry( geometry );
  admmtotalvariation->SetAlpha( 100 );
  admmtotalvariation->SetBeta( 1000 );
  admmtotalvariation->SetAL_iterations( 3 );
  admmtotalvariation->SetCG_iterations( 2 );

  // In all cases except CUDA, use the Joseph forward projector
  admmtotalvariation->SetForwardProjectionFilter(0);

  std::cout << "\n\n****** Case 1: Voxel-Based Backprojector ******" << std::endl;

  admmtotalvariation->SetBackProjectionFilter( 0 );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( admmtotalvariation->Update() );

  CheckImageQuality<OutputImageType>(admmtotalvariation->GetOutput(), dsl->GetOutput(), 0.05, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Joseph Backprojector ******" << std::endl;

  admmtotalvariation->SetBackProjectionFilter( 1 );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( admmtotalvariation->Update() );

  CheckImageQuality<OutputImageType>(admmtotalvariation->GetOutput(), dsl->GetOutput(), 0.05, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 3: CUDA Voxel-Based Backprojector and CUDA Forward projector ******" << std::endl;

  admmtotalvariation->SetForwardProjectionFilter( 2 );
  admmtotalvariation->SetBackProjectionFilter( 2 );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( admmtotalvariation->Update() );

  CheckImageQuality<OutputImageType>(admmtotalvariation->GetOutput(), dsl->GetOutput(), 0.05, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  std::cout << "\n\n****** Voxel-Based Backprojector and gating ******" << std::endl;

  admmtotalvariation->SetForwardProjectionFilter(0); // Joseph
  admmtotalvariation->SetBackProjectionFilter( 0 ); // Voxel based

  // Generate arbitrary gating weights (select every third projection)
  typedef rtk::PhaseGatingImageFilter<OutputImageType> PhaseGatingFilterType;
  PhaseGatingFilterType::Pointer phaseGating = PhaseGatingFilterType::New();
#if FAST_TESTS_NO_CHECKS
  phaseGating->SetPhasesFileName(std::string(RTK_DATA_ROOT) +
                           std::string("/Input/Phases/phases_3projs.txt"));
#else
  phaseGating->SetPhasesFileName(std::string(RTK_DATA_ROOT) +
                           std::string("/Input/Phases/phases.txt"));
#endif
  phaseGating->SetGatingWindowWidth(0.20);
  phaseGating->SetGatingWindowShape(0); // Rectangular
  phaseGating->SetGatingWindowCenter(0.70);
  phaseGating->SetInputProjectionStack(rei->GetOutput());
  phaseGating->SetInputGeometry(geometry);
  phaseGating->Update();

  admmtotalvariation->SetInput(1, phaseGating->GetOutput());
  admmtotalvariation->SetGeometry( phaseGating->GetOutputGeometry() );
  admmtotalvariation->SetGatingWeights( phaseGating->GetGatingWeightsOnSelectedProjections() );

  TRY_AND_EXIT_ON_ITK_EXCEPTION( admmtotalvariation->Update() );

  CheckImageQuality<OutputImageType>(admmtotalvariation->GetOutput(), dsl->GetOutput(), 0.05, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;


  return EXIT_SUCCESS;
}
