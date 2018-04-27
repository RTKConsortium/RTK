#include <itkImageRegionConstIterator.h>

#include "rtkTest.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"

#ifdef RTK_USE_CUDA
  #include "itkCudaImage.h"
#endif
#include "rtkOSEMConeBeamReconstructionFilter.h"

/**
 * \file rtkosemtest.cxx
 *
 * \brief Functional test for OSEM reconstruction
 *
 * This test generates the projections of an ellipsoid and reconstructs the CT
 * image using the OSEM algorithm with different backprojectors (Voxel-Based,
 * Joseph and CUDA Voxel-Based). The generated results are compared to the
 * expected results (analytical calculation).
 *
 * \author Simon Rit and Marc Vila
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
  ConstantImageSourceType::Pointer volumeSource  = ConstantImageSourceType::New();
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
  volumeSource->SetOrigin( origin );
  volumeSource->SetSpacing( spacing );
  volumeSource->SetSize( size );
  volumeSource->SetConstant( 1. );

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

  // OSEM reconstruction filtering
  typedef rtk::OSEMConeBeamReconstructionFilter< OutputImageType > OSEMType;
  OSEMType::Pointer osem = OSEMType::New();
  osem->SetInput(0, volumeSource->GetOutput() );
  osem->SetInput(1, rei->GetOutput());
  osem->SetGeometry( geometry );

  std::cout << "\n\n****** Case 1: Voxel-Based Backprojector, ML-EM 7 iterations ******" << std::endl;

  osem->SetNumberOfIterations(7);
  osem->SetBackProjectionFilter( 0 ); // Voxel based
  osem->SetForwardProjectionFilter( 0 ); // Joseph
  osem->SetNumberOfProjectionsPerSubset(NumberOfProjectionImages);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( osem->Update() );

  CheckImageQuality<OutputImageType>(osem->GetOutput(), dsl->GetOutput(), 0.032, 28.0, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Joseph-Based Backprojector, OS-EM with 50 projections per subset and 4 iterations******" << std::endl;

  osem->SetNumberOfIterations(4);
  osem->SetBackProjectionFilter( 1 ); // Joseph based
  osem->SetForwardProjectionFilter( 0 ); // Joseph
  osem->SetNumberOfProjectionsPerSubset(50);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( osem->Update() );

  CheckImageQuality<OutputImageType>(osem->GetOutput(), dsl->GetOutput(), 0.032, 28.0, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 3: Voxel-Based Backprojector, OS-EM with 50 projections per subset and 4 iterations******" << std::endl;

  osem->SetNumberOfIterations(4);
  osem->SetBackProjectionFilter( 0 ); // Voxel-Based
  osem->SetForwardProjectionFilter( 0 ); // Joseph
  osem->SetNumberOfProjectionsPerSubset(50);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( osem->Update() );

  CheckImageQuality<OutputImageType>(osem->GetOutput(), dsl->GetOutput(), 0.032, 28.6, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 4: CUDA Voxel-Based Backprojector ******" << std::endl;

  osem->SetBackProjectionFilter( 2 ); // Cuda voxel based
  osem->SetForwardProjectionFilter( 1 ); // Cuda ray cast
  TRY_AND_EXIT_ON_ITK_EXCEPTION( osem->Update() );

  CheckImageQuality<OutputImageType>(osem->GetOutput(), dsl->GetOutput(), 0.032, 28.6, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif


  return EXIT_SUCCESS;
}
