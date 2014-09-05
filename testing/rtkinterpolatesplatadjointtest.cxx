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

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;

  typedef itk::Image< OutputPixelType, Dimension > VolumeType;
  typedef itk::Image< OutputPixelType, Dimension + 1 > VolumeSeriesType;

#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfSlices = 2;
#else
  const unsigned int NumberOfSlices = 64;
#endif


  // Random image sources
  typedef itk::RandomImageSource< VolumeType > RandomVolumeSourceType;
  RandomVolumeSourceType::Pointer randomVolumeSource  = RandomVolumeSourceType::New();

  typedef itk::RandomImageSource< VolumeSeriesType > RandomVolumeSeriesSourceType;
  RandomVolumeSeriesSourceType::Pointer randomVolumeSeriesSource  = RandomVolumeSeriesSourceType::New();

  // Constant sources
  typedef rtk::ConstantImageSource< VolumeType > ConstantVolumeSourceType;
  ConstantVolumeSourceType::Pointer constantVolumeSource  = ConstantVolumeSourceType::New();

  typedef rtk::ConstantImageSource< VolumeSeriesType > ConstantVolumeSeriesSourceType;
  ConstantVolumeSeriesSourceType::Pointer constantVolumeSeriesSource  = ConstantVolumeSeriesSourceType::New();
  
  // Volume metadata
  VolumeSeriesType::PointType fourDOrigin;
  VolumeSeriesType::SizeType fourDSize;
  VolumeSeriesType::SpacingType fourDSpacing;

  fourDOrigin[0] = -127.;
  fourDOrigin[1] = -127.;
  fourDOrigin[2] = -127.;
  fourDOrigin[3] = 0.;
#if FAST_TESTS_NO_CHECKS
  fourDSize[0] = 2;
  fourDSize[1] = 2;
  fourDSize[2] = NumberOfSlices;
  fourDSize[3] = 2;
  fourDSpacing[0] = 252.;
  fourDSpacing[1] = 252.;
  fourDSpacing[2] = 252.;
  fourDSpacing[3] = 1.;
#else
  fourDSize[0] = 64;
  fourDSize[1] = 64;
  fourDSize[2] = NumberOfSlices;
  fourDSize[3] = 5;
  fourDSpacing[0] = 4.;
  fourDSpacing[1] = 4.;
  fourDSpacing[2] = 4.;
  fourDSpacing[3] = 1.;
#endif
  randomVolumeSeriesSource->SetOrigin( fourDOrigin );
  randomVolumeSeriesSource->SetSpacing( fourDSpacing );
  randomVolumeSeriesSource->SetSize( fourDSize );
  randomVolumeSeriesSource->SetMin( 0. );
  randomVolumeSeriesSource->SetMax( 1. );

  constantVolumeSeriesSource->SetOrigin( fourDOrigin );
  constantVolumeSeriesSource->SetSpacing( fourDSpacing );
  constantVolumeSeriesSource->SetSize( fourDSize );
  constantVolumeSeriesSource->SetConstant( 0. );

  // Volume metadata
  VolumeType::PointType origin;
  VolumeType::SizeType size;
  VolumeType::SpacingType spacing;

  origin[0] = -255.;
  origin[1] = -255.;
  origin[2] = -255.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = NumberOfSlices;
  spacing[0] = 504.;
  spacing[1] = 504.;
  spacing[2] = 504.;
#else
  size[0] = 64;
  size[1] = 64;
  size[2] = NumberOfSlices;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
#endif
  randomVolumeSource->SetOrigin( origin );
  randomVolumeSource->SetSpacing( spacing );
  randomVolumeSource->SetSize( size );
  randomVolumeSource->SetMin( 0. );
  randomVolumeSource->SetMax( 1. );

  constantVolumeSource->SetOrigin( origin );
  constantVolumeSource->SetSpacing( spacing );
  constantVolumeSource->SetSize( size );
  constantVolumeSource->SetConstant( 0. );

  // Update all sources
  TRY_AND_EXIT_ON_ITK_EXCEPTION( randomVolumeSeriesSource->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( constantVolumeSeriesSource->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( randomVolumeSource->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( constantVolumeSource->Update() );

  // Read the phases file
  rtk::PhasesToInterpolationWeights::Pointer phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(std::string(RTK_DATA_ROOT) +
                           std::string("/Input/Phases/phases_slow.txt"));
  phaseReader->SetNumberOfReconstructedFrames( fourDSize[3] );
  phaseReader->Update();

  std::cout << "\n\n****** 4D to 3D (interpolation) ******" << std::endl;

  typedef rtk::InterpolatorWithKnownWeightsImageFilter<VolumeType, VolumeSeriesType> InterpolateFilterType;
  InterpolateFilterType::Pointer interp = InterpolateFilterType::New();
  interp->SetInputVolume(constantVolumeSource->GetOutput());
  interp->SetInputVolumeSeries(randomVolumeSeriesSource->GetOutput());
  interp->SetWeights(phaseReader->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION( interp->Update() );

  std::cout << "\n\n****** 3D to 4D (splat) ******" << std::endl;

  typedef rtk::SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType> SplatFilterType;
  SplatFilterType::Pointer splat = SplatFilterType::New();
  splat->SetInputVolumeSeries(constantVolumeSeriesSource->GetOutput());
  splat->SetInputVolume(randomVolumeSource->GetOutput());
  splat->SetWeights(phaseReader->GetOutput());

// If ITK version < 4.4, try to force the splat filter to use several threads
// Splat filter should display a warning and revert to single threading
#if !(ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4))
  splat->SetNumberOfThreads(2);
#endif

  TRY_AND_EXIT_ON_ITK_EXCEPTION( splat->Update() );

  CheckScalarProducts<VolumeSeriesType, VolumeType>(randomVolumeSeriesSource->GetOutput(), splat->GetOutput(), randomVolumeSource->GetOutput(), interp->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
