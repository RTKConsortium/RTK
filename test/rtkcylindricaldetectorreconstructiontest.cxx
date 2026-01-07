#include "rtkTest.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkConjugateGradientConeBeamReconstructionFilter.h"

#ifdef USE_CUDA
#  include "itkCudaImage.h"
#endif
#include "itkTestingMacros.h"

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

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;

#ifdef USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 180;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto tomographySource = ConstantImageSourceType::New();
  auto origin = itk::MakePoint(-127., -127., -127.);
#if FAST_TESTS_NO_CHECKS
  auto spacing = itk::MakeVector(252., 252., 252.);
  auto size = itk::MakeSize(2, 2, 2);
#else
  auto spacing = itk::MakeVector(4., 4., 4.);
  auto size = itk::MakeSize(64, 64, 64);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  auto projectionsSource = ConstantImageSourceType::New();
  origin = itk::MakePoint(-255., -255., -255.);
#if FAST_TESTS_NO_CHECKS
  spacing = itk::MakeVector(504., 504., 504.);
  size = itk::MakeSize(2, 2, NumberOfProjectionImages);
#else
  spacing = itk::MakeVector(8., 8., 8.);
  size = itk::MakeSize(64, 64, NumberOfProjectionImages);
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  // Geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto geometry = GeometryType::New();
  geometry->SetRadiusCylindricalDetector(200);
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);

  // Create ellipsoid PROJECTIONS
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType>;
  REIType::Pointer rei;

  rei = REIType::New();
  rei->SetAngle(0.);
  rei->SetDensity(1.);
  rei->SetCenter(itk::MakePoint(0., 0., 0.));
  rei->SetAxis(itk::MakeVector(90., 90., 90.));

  rei->SetInput(projectionsSource->GetOutput());
  rei->SetGeometry(geometry);

  // Update
  TRY_AND_EXIT_ON_ITK_EXCEPTION(rei->Update());

  // Create REFERENCE object (3D ellipsoid).
  auto dsl = rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType>::New();
  dsl->SetInput(tomographySource->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update())

  // Define weights (for weighted least squares)
  auto uniformWeightsSource = ConstantImageSourceType::New();
  uniformWeightsSource->SetInformationFromImage(projectionsSource->GetOutput());
  uniformWeightsSource->SetConstant(1.0);

  // ConjugateGradient reconstruction filtering
  using ConjugateGradientType = rtk::ConjugateGradientConeBeamReconstructionFilter<OutputImageType>;
  auto conjugategradient = ConjugateGradientType::New();
  conjugategradient->SetInput(tomographySource->GetOutput());
  conjugategradient->SetInput(1, rei->GetOutput());
  conjugategradient->SetInputWeights(uniformWeightsSource->GetOutput());
  conjugategradient->SetGeometry(geometry);
  conjugategradient->SetNumberOfIterations(5);
  conjugategradient->SetDisableDisplacedDetectorFilter(true);

  std::cout << "\n\n****** Case 1: Joseph forward and back projectors ******" << std::endl;

  conjugategradient->SetForwardProjectionFilter(ConjugateGradientType::FP_JOSEPH);
  conjugategradient->SetBackProjectionFilter(ConjugateGradientType::BP_JOSEPH);
#if defined(RTK_USE_CUDA) && !defined(USE_CUDA)
  ITK_TRY_EXPECT_EXCEPTION(conjugategradient->Update());
  conjugategradient->SetCudaConjugateGradient(false);
#endif
  TRY_AND_EXIT_ON_ITK_EXCEPTION(conjugategradient->Update());

  CheckImageQuality<OutputImageType>(conjugategradient->GetOutput(), dsl->GetOutput(), 0.08, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 2: CUDA ray cast forward and back projectors ******" << std::endl;

  conjugategradient->SetCudaConjugateGradient(true);
  conjugategradient->SetForwardProjectionFilter(ConjugateGradientType::FP_CUDARAYCAST);
  conjugategradient->SetBackProjectionFilter(ConjugateGradientType::BP_CUDARAYCAST);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(conjugategradient->Update());

  CheckImageQuality<OutputImageType>(conjugategradient->GetOutput(), dsl->GetOutput(), 0.08, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  std::cout << "\n\n****** Case 3: Joseph forward projector, voxel-based back projector ******" << std::endl;

  // Only works with a cylindrical detector centered on the source, therefore a new geometry has to be simulated
  geometry = GeometryType::New();
  geometry->SetRadiusCylindricalDetector(1200);
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);

  conjugategradient->SetGeometry(geometry);
  conjugategradient->SetForwardProjectionFilter(ConjugateGradientType::FP_JOSEPH);
  conjugategradient->SetBackProjectionFilter(ConjugateGradientType::BP_VOXELBASED);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(conjugategradient->Update());

  CheckImageQuality<OutputImageType>(conjugategradient->GetOutput(), dsl->GetOutput(), 0.08, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 4: CUDA ray cast forward projection, CUDA voxel based back projection ******"
            << std::endl;

  conjugategradient->SetForwardProjectionFilter(ConjugateGradientType::FP_CUDARAYCAST);
  conjugategradient->SetBackProjectionFilter(ConjugateGradientType::BP_CUDAVOXELBASED);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(conjugategradient->Update());

  CheckImageQuality<OutputImageType>(conjugategradient->GetOutput(), dsl->GetOutput(), 0.08, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif


  return EXIT_SUCCESS;
}
