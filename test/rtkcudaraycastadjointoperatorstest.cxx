#include "rtkMacro.h"
#include "rtkTest.h"
#include "itkRandomImageSource.h"
#include "rtkConstantImageSource.h"
#include "rtkCudaRayCastBackProjectionImageFilter.h"
#include "rtkCudaForwardProjectionImageFilter.h"

/**
 * \file rtkcudaraycastadjointoperatorstest.cxx
 *
 * \brief Tests whether CUDA ray cast forward and back projectors are matched
 *
 * This test generates a random volume "v" and a random set of projections "p",
 * and compares the scalar products <Rv , p> and <v, R* p>, where R is the
 * CUDA ray cast forward projector and R* is the CUDA ray cast back projector. If R* is indeed
 * the adjoint of R, these scalar products are equal.
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


  // Random image sources
  using RandomImageSourceType = itk::RandomImageSource<OutputImageType>;
  RandomImageSourceType::Pointer randomVolumeSource = RandomImageSourceType::New();
  RandomImageSourceType::Pointer randomProjectionsSource = RandomImageSourceType::New();

  // Constant sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::Pointer constantVolumeSource = ConstantImageSourceType::New();
  ConstantImageSourceType::Pointer constantProjectionsSource = ConstantImageSourceType::New();

  // Volume metadata
  auto origin = itk::MakePoint(-127., -127., -127.);
#if FAST_TESTS_NO_CHECKS
  auto spacing = itk::MakeVector(252., 252., 252.);
  auto size = itk::MakeSize(2, 2, 2);
#else
  auto spacing = itk::MakeVector(4., 4., 4.);
  auto size = itk::MakeSize(64, 64, 64);
#endif
  randomVolumeSource->SetOrigin(origin);
  randomVolumeSource->SetSpacing(spacing);
  randomVolumeSource->SetSize(size);
  randomVolumeSource->SetMin(0.);
  randomVolumeSource->SetMax(1.);

  constantVolumeSource->SetOrigin(origin);
  constantVolumeSource->SetSpacing(spacing);
  constantVolumeSource->SetSize(size);
  constantVolumeSource->SetConstant(0.);

  // Projections metadata
  origin = itk::MakePoint(-255., -255., -255.);
#if FAST_TESTS_NO_CHECKS
  spacing = itk::MakeVector(504., 504., 504.);
  size = itk::MakeSize(2, 2, NumberOfProjectionImages);
#else
  spacing = itk::MakeVector(8., 8., 8.);
  size = itk::MakeSize(64, 64, NumberOfProjectionImages);
#endif
  randomProjectionsSource->SetOrigin(origin);
  randomProjectionsSource->SetSpacing(spacing);
  randomProjectionsSource->SetSize(size);
  randomProjectionsSource->SetMin(0.);
  randomProjectionsSource->SetMax(100.);

  constantProjectionsSource->SetOrigin(origin);
  constantProjectionsSource->SetSpacing(spacing);
  constantProjectionsSource->SetSize(size);
  constantProjectionsSource->SetConstant(0.);

  // Update all sources
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomVolumeSource->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantVolumeSource->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomProjectionsSource->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantProjectionsSource->Update());

  // Geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);

  std::cout << "\n\n****** CUDA ray cast Forward projector, flat panel detector ******" << std::endl;

  using ForwardProjectorType = rtk::CudaForwardProjectionImageFilter<OutputImageType, OutputImageType>;
  ForwardProjectorType::Pointer fw = ForwardProjectorType::New();
  fw->SetInput(0, constantProjectionsSource->GetOutput());
  fw->SetInput(1, randomVolumeSource->GetOutput());
  fw->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(fw->Update());

  std::cout << "\n\n****** CUDA ray cast Back projector, flat panel detector ******" << std::endl;

  using BackProjectorType = rtk::CudaRayCastBackProjectionImageFilter;
  BackProjectorType::Pointer bp = BackProjectorType::New();
  bp->SetInput(0, constantVolumeSource->GetOutput());
  bp->SetInput(1, randomProjectionsSource->GetOutput());
  bp->SetGeometry(geometry.GetPointer());
  bp->SetNormalize(false);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(bp->Update());

  CheckScalarProducts<OutputImageType, OutputImageType>(
    randomVolumeSource->GetOutput(), bp->GetOutput(), randomProjectionsSource->GetOutput(), fw->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  // Start over with cylindrical detector
  geometry->SetRadiusCylindricalDetector(200);
  std::cout << "\n\n****** CUDA ray cast Forward projector, cylindrical detector ******" << std::endl;

  fw->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(fw->Update());

  std::cout << "\n\n****** CUDA ray cast Back projector, cylindrical detector ******" << std::endl;

  bp->SetGeometry(geometry.GetPointer());
  bp->SetNormalize(false);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(bp->Update());

  CheckScalarProducts<OutputImageType, OutputImageType>(
    randomVolumeSource->GetOutput(), bp->GetOutput(), randomProjectionsSource->GetOutput(), fw->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
