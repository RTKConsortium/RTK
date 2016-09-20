#include "rtkTest.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkRegularizedConjugateGradientConeBeamReconstructionFilter.h"

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 6)
  #include <itkAdditiveGaussianNoiseImageFilter.h>
#endif

#ifdef USE_CUDA
  #include "itkCudaImage.h"
#endif

/**
 * \file rtkregularizedconjugategradienttest.cxx
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
#else
  typedef itk::Image< OutputPixelType, Dimension >     OutputImageType;
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

  // Create the weights map
  ConstantImageSourceType::Pointer uniformWeightsSource = ConstantImageSourceType::New();
  uniformWeightsSource->SetInformationFromImage(projectionsSource->GetOutput());
  uniformWeightsSource->SetConstant(1.0);

  // Regularized CG reconstruction filter
  typedef rtk::RegularizedConjugateGradientConeBeamReconstructionFilter<OutputImageType>   RegularizedCGType;
  RegularizedCGType::Pointer regularizedConjugateGradient = RegularizedCGType::New();
  regularizedConjugateGradient->SetInputVolume(tomographySource->GetOutput() );
  regularizedConjugateGradient->SetInputProjectionStack(rei->GetOutput());
  regularizedConjugateGradient->SetInputWeights( uniformWeightsSource->GetOutput());
  regularizedConjugateGradient->SetPreconditioned(false);
  regularizedConjugateGradient->SetGeometry( geometry );
  regularizedConjugateGradient->SetMainLoop_iterations( 2 );
  regularizedConjugateGradient->SetCudaConjugateGradient(false);

  regularizedConjugateGradient->SetGammaTV(1);
  regularizedConjugateGradient->SetTV_iterations( 3 );

  regularizedConjugateGradient->SetSoftThresholdWavelets(0.1);
  regularizedConjugateGradient->SetOrder(3);
  regularizedConjugateGradient->SetNumberOfLevels(3);

  // In all cases except CUDA, use the Joseph forward projector and Voxel-based back projector
  regularizedConjugateGradient->SetForwardProjectionFilter(0);
  regularizedConjugateGradient->SetBackProjectionFilter( 0 );

  std::cout << "\n\n****** Case 1: Positivity + TV regularization ******" << std::endl;

  regularizedConjugateGradient->SetPerformPositivity(true);
  regularizedConjugateGradient->SetPerformTVSpatialDenoising(true);
  regularizedConjugateGradient->SetPerformWaveletsSpatialDenoising(false);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( regularizedConjugateGradient->Update() );

  CheckImageQuality<OutputImageType>(regularizedConjugateGradient->GetOutput(), dsl->GetOutput(), 0.05, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Wavelets ******" << std::endl;

  regularizedConjugateGradient->SetPerformPositivity(false);
  regularizedConjugateGradient->SetPerformTVSpatialDenoising(false);
  regularizedConjugateGradient->SetPerformWaveletsSpatialDenoising(true);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( regularizedConjugateGradient->Update() );

  CheckImageQuality<OutputImageType>(regularizedConjugateGradient->GetOutput(), dsl->GetOutput(), 0.05, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 3: CUDA Voxel-Based Backprojector and CUDA Forward projector, all regularization steps on ******" << std::endl;

  regularizedConjugateGradient->SetForwardProjectionFilter( 2 );
  regularizedConjugateGradient->SetBackProjectionFilter( 2 );
  regularizedConjugateGradient->SetCudaConjugateGradient(true);
  regularizedConjugateGradient->SetPerformPositivity(true);
  regularizedConjugateGradient->SetPerformTVSpatialDenoising(true);
  regularizedConjugateGradient->SetPerformWaveletsSpatialDenoising(true);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( regularizedConjugateGradient->Update() );

  CheckImageQuality<OutputImageType>(regularizedConjugateGradient->GetOutput(), dsl->GetOutput(), 0.05, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 6)
  std::cout << "\n\n****** Image-domain sparsity ******" << std::endl;

  // Replace the ellise with a very small one
  semiprincipalaxis.Fill(9.);
  rei->SetAxis(semiprincipalaxis);
  dsl->SetAxis(semiprincipalaxis);

  // Add gaussian noise on the projections
  typedef itk::AdditiveGaussianNoiseImageFilter<OutputImageType>      GaussianNoiseFilterType;
  GaussianNoiseFilterType::Pointer gaussian = GaussianNoiseFilterType::New();
  gaussian->SetStandardDeviation(1);
  gaussian->SetMean(0.);
  gaussian->SetInput(rei->GetOutput());

  regularizedConjugateGradient->SetInputProjectionStack(gaussian->GetOutput());
  regularizedConjugateGradient->SetPerformPositivity(false);
  regularizedConjugateGradient->SetPerformTVSpatialDenoising(false);
  regularizedConjugateGradient->SetPerformWaveletsSpatialDenoising(false);
  regularizedConjugateGradient->SetPerformSoftThresholdOnImage(true);
  regularizedConjugateGradient->SetSoftThresholdOnImage(0.01);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( regularizedConjugateGradient->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() );

  CheckImageQuality<OutputImageType>(regularizedConjugateGradient->GetOutput(), dsl->GetOutput(), 0.004, 47, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  return EXIT_SUCCESS;
}
