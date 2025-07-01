#include "rtkTest.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkRegularizedConjugateGradientConeBeamReconstructionFilter.h"

#include <itkAdditiveGaussianNoiseImageFilter.h>

#ifdef USE_CUDA
#  include "itkCudaImage.h"
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

int
main(int, char **)
{
  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

#ifdef USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 90;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto tomographySource = ConstantImageSourceType::New();
  auto origin = itk::MakePoint(-127., -127., -127.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(2, 2, 2);
  auto spacing = itk::MakeVector(252., 252., 252.);
#else
  auto size = itk::MakeSize(64, 64, 64);
  auto spacing = itk::MakeVector(4., 4., 4.);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  auto projectionsSource = ConstantImageSourceType::New();
  origin = itk::MakePoint(-255., -255., -255.);
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(2, 2, NumberOfProjectionImages);
  spacing = itk::MakeVector(504., 504., 504.);
#else
  size = itk::MakeSize(64, 64, NumberOfProjectionImages);
  spacing = itk::MakeVector(8., 8., 8.);
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

  // Create ellipsoid PROJECTIONS
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType>;
  REIType::Pointer rei;

  rei = REIType::New();
  rei->SetAngle(0.);
  rei->SetDensity(1.);
  rei->SetCenter(itk::MakeVector(0., 0., 0.));
  rei->SetAxis(itk::MakeVector(90., 90., 90.));
  rei->SetInput(projectionsSource->GetOutput());
  rei->SetGeometry(geometry);

  // Update
  TRY_AND_EXIT_ON_ITK_EXCEPTION(rei->Update());

  // Create REFERENCE object (3D ellipsoid).
  using DEType = rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType>;
  auto dsl = DEType::New();
  dsl->SetInput(tomographySource->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update())

  // Create the weights map
  auto uniformWeightsSource = ConstantImageSourceType::New();
  uniformWeightsSource->SetInformationFromImage(projectionsSource->GetOutput());
  uniformWeightsSource->SetConstant(1.0);

  // Regularized CG reconstruction filter
  using RegularizedCGType = rtk::RegularizedConjugateGradientConeBeamReconstructionFilter<OutputImageType>;
  auto regularizedConjugateGradient = RegularizedCGType::New();
  regularizedConjugateGradient->SetInputVolume(tomographySource->GetOutput());
  regularizedConjugateGradient->SetInputProjectionStack(rei->GetOutput());
  regularizedConjugateGradient->SetInputWeights(uniformWeightsSource->GetOutput());
  regularizedConjugateGradient->SetPreconditioned(false);
  regularizedConjugateGradient->SetGeometry(geometry);
  regularizedConjugateGradient->SetMainLoop_iterations(2);
  regularizedConjugateGradient->SetCudaConjugateGradient(false);

  regularizedConjugateGradient->SetGammaTV(1);
  regularizedConjugateGradient->SetTV_iterations(3);

  regularizedConjugateGradient->SetSoftThresholdWavelets(0.1);
  regularizedConjugateGradient->SetOrder(3);
  regularizedConjugateGradient->SetNumberOfLevels(3);

  // In all cases except CUDA, use the Joseph forward projector and Voxel-based back projector
  regularizedConjugateGradient->SetForwardProjectionFilter(RegularizedCGType::FP_JOSEPH);
  regularizedConjugateGradient->SetBackProjectionFilter(RegularizedCGType::BP_VOXELBASED);

  std::cout << "\n\n****** Case 1: Positivity + TV regularization ******" << std::endl;

  regularizedConjugateGradient->SetPerformPositivity(true);
  regularizedConjugateGradient->SetPerformTVSpatialDenoising(true);
  regularizedConjugateGradient->SetPerformWaveletsSpatialDenoising(false);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(regularizedConjugateGradient->Update());

  CheckImageQuality<OutputImageType>(regularizedConjugateGradient->GetOutput(), dsl->GetOutput(), 0.05, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Wavelets ******" << std::endl;

  regularizedConjugateGradient->SetPerformPositivity(false);
  regularizedConjugateGradient->SetPerformTVSpatialDenoising(false);
  regularizedConjugateGradient->SetPerformWaveletsSpatialDenoising(true);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(regularizedConjugateGradient->Update());

  CheckImageQuality<OutputImageType>(regularizedConjugateGradient->GetOutput(), dsl->GetOutput(), 0.05, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 3: CUDA Voxel-Based Backprojector and CUDA Forward projector, all regularization steps "
               "on ******"
            << std::endl;

  regularizedConjugateGradient->SetForwardProjectionFilter(RegularizedCGType::FP_CUDARAYCAST);
  regularizedConjugateGradient->SetBackProjectionFilter(RegularizedCGType::BP_CUDAVOXELBASED);
  regularizedConjugateGradient->SetCudaConjugateGradient(true);
  regularizedConjugateGradient->SetPerformPositivity(true);
  regularizedConjugateGradient->SetPerformTVSpatialDenoising(true);
  regularizedConjugateGradient->SetPerformWaveletsSpatialDenoising(true);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(regularizedConjugateGradient->Update());

  CheckImageQuality<OutputImageType>(regularizedConjugateGradient->GetOutput(), dsl->GetOutput(), 0.05, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  std::cout << "\n\n****** Image-domain sparsity ******" << std::endl;

  // Replace the ellise with a very small one
  rei->SetAxis(itk::MakeVector(9., 9., 9.));
  dsl->SetAxis(itk::MakeVector(9., 9., 9.));

  // Add gaussian noise on the projections
  using GaussianNoiseFilterType = itk::AdditiveGaussianNoiseImageFilter<OutputImageType>;
  auto gaussian = GaussianNoiseFilterType::New();
  gaussian->SetStandardDeviation(1);
  gaussian->SetMean(0.);
  gaussian->SetInput(rei->GetOutput());

  regularizedConjugateGradient->SetInputProjectionStack(gaussian->GetOutput());
  regularizedConjugateGradient->SetPerformPositivity(false);
  regularizedConjugateGradient->SetPerformTVSpatialDenoising(false);
  regularizedConjugateGradient->SetPerformWaveletsSpatialDenoising(false);
  regularizedConjugateGradient->SetPerformSoftThresholdOnImage(true);
  regularizedConjugateGradient->SetSoftThresholdOnImage(0.01);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(regularizedConjugateGradient->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update());

  CheckImageQuality<OutputImageType>(regularizedConjugateGradient->GetOutput(), dsl->GetOutput(), 0.004, 47, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
