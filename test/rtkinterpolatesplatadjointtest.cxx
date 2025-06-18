#include "rtkTest.h"
#include "itkRandomImageSource.h"
#include "rtkConstantImageSource.h"
#include "rtkInterpolatorWithKnownWeightsImageFilter.h"
#include "rtkSplatWithKnownWeightsImageFilter.h"
#include "rtkPhasesToInterpolationWeights.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkMacro.h"

#include <itkImageFileReader.h>

/**
 * \file rtkinterpolatesplatadjointtest.cxx
 *
 * \brief Tests whether InterpolatorWithKnownWeightsImageFilter and
 * SplatWithKnownWeightsImageFilter are the adjoint of one another.
 *
 * This test generates a random 4D volume "f" and a random 3D volume "v",
 * and compares the scalar products <Sf , v> and <f, S* v>, where S is the
 * interpolation operator and S* is the splat operator. If S* is indeed
 * the adjoint of S, these scalar products are equal.
 *
 * \author Cyril Mory
 */

int
main(int argc, char * argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " phases.txt" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;

  using VolumeType = itk::Image<OutputPixelType, Dimension>;
  using VolumeSeriesType = itk::Image<OutputPixelType, Dimension + 1>;

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfSlices = 2;
#else
  constexpr unsigned int NumberOfSlices = 64;
#endif


  // Random image sources
  using RandomVolumeSourceType = itk::RandomImageSource<VolumeType>;
  RandomVolumeSourceType::Pointer randomVolumeSource = RandomVolumeSourceType::New();

  using RandomVolumeSeriesSourceType = itk::RandomImageSource<VolumeSeriesType>;
  RandomVolumeSeriesSourceType::Pointer randomVolumeSeriesSource = RandomVolumeSeriesSourceType::New();

  // Constant sources
  using ConstantVolumeSourceType = rtk::ConstantImageSource<VolumeType>;
  ConstantVolumeSourceType::Pointer constantVolumeSource = ConstantVolumeSourceType::New();

  using ConstantVolumeSeriesSourceType = rtk::ConstantImageSource<VolumeSeriesType>;
  ConstantVolumeSeriesSourceType::Pointer constantVolumeSeriesSource = ConstantVolumeSeriesSourceType::New();

  // Volume metadata
  auto fourDOrigin = itk::MakePoint(-127., -127., -127., 0.);
#if FAST_TESTS_NO_CHECKS
  auto fourDSize = itk::MakeSize(2, 2, NumberOfSlices, 2);
  auto fourDSpacing = itk::MakeVector(252., 252., 252., 1.);
#else
  auto fourDSize = itk::MakeSize(64, 64, NumberOfSlices, 5);
  auto fourDSpacing = itk::MakeVector(4., 4., 4., 1.);
#endif
  randomVolumeSeriesSource->SetOrigin(fourDOrigin);
  randomVolumeSeriesSource->SetSpacing(fourDSpacing);
  randomVolumeSeriesSource->SetSize(fourDSize);
  randomVolumeSeriesSource->SetMin(0.);
  randomVolumeSeriesSource->SetMax(1.);

  constantVolumeSeriesSource->SetOrigin(fourDOrigin);
  constantVolumeSeriesSource->SetSpacing(fourDSpacing);
  constantVolumeSeriesSource->SetSize(fourDSize);
  constantVolumeSeriesSource->SetConstant(0.);

  // Volume metadata
  auto origin = itk::MakePoint(-255., -255., -255.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(2, 2, NumberOfSlices);
  auto spacing = itk::MakeVector(504., 504., 504.);
#else
  auto size = itk::MakeSize(64, 64, NumberOfSlices);
  auto spacing = itk::MakeVector(8., 8., 8.);
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

  // Update all sources
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomVolumeSeriesSource->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantVolumeSeriesSource->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomVolumeSource->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantVolumeSource->Update());

  // Read the phases file
  rtk::PhasesToInterpolationWeights::Pointer phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(argv[1]);
  phaseReader->SetNumberOfReconstructedFrames(fourDSize[3]);
  phaseReader->Update();

  std::cout << "\n\n****** 4D to 3D (interpolation) ******" << std::endl;

  using InterpolateFilterType = rtk::InterpolatorWithKnownWeightsImageFilter<VolumeType, VolumeSeriesType>;
  InterpolateFilterType::Pointer interp = InterpolateFilterType::New();
  interp->SetInputVolume(constantVolumeSource->GetOutput());
  interp->SetInputVolumeSeries(randomVolumeSeriesSource->GetOutput());
  interp->SetWeights(phaseReader->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(interp->Update());

  std::cout << "\n\n****** 3D to 4D (splat) ******" << std::endl;

  using SplatFilterType = rtk::SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>;
  SplatFilterType::Pointer splat = SplatFilterType::New();
  splat->SetInputVolumeSeries(constantVolumeSeriesSource->GetOutput());
  splat->SetInputVolume(randomVolumeSource->GetOutput());
  splat->SetWeights(phaseReader->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(splat->Update());

  CheckScalarProducts<VolumeSeriesType, VolumeType>(
    randomVolumeSeriesSource->GetOutput(), splat->GetOutput(), randomVolumeSource->GetOutput(), interp->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
