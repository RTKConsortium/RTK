#include "rtkTest.h"
#include "itkRandomImageSource.h"
#include "rtkConstantImageSource.h"
#include "rtkProjectionStackToFourDImageFilter.h"
#include "rtkFourDToProjectionStackImageFilter.h"
#include "rtkPhasesToInterpolationWeights.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkMacro.h"

#include <itkImageFileReader.h>

/**
 * \file rtkfourdadjointoperatorstest.cxx
 *
 * \brief Tests whether ProjectionStackToFourDImageFilter and FourDToProjectionStackImageFilter
 * are the adjoint of one another.
 *
 * This test generates a random 4D volume "v" and a random set of projections "p",
 * and compares the scalar products <Rv , p> and <v, R* p>, where R is the
 * FourDToProjectionStack filter and R* is the ProjectionStackToFourD. If R* is indeed
 * the adjoint of R, these scalar products are equal.
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

#ifdef RTK_USE_CUDA
  using ProjectionStackType = itk::CudaImage<OutputPixelType, Dimension>;
  using VolumeSeriesType = itk::CudaImage<OutputPixelType, Dimension + 1>;
#else
  using ProjectionStackType = itk::Image<OutputPixelType, Dimension>;
  using VolumeSeriesType = itk::Image<OutputPixelType, Dimension + 1>;
#endif

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 64;
#endif


  // Random image sources
  auto randomProjectionStackSource = itk::RandomImageSource<ProjectionStackType>::New();

  auto randomVolumeSeriesSource = itk::RandomImageSource<VolumeSeriesType>::New();

  // Constant sources
  auto constantProjectionStackSource = rtk::ConstantImageSource<ProjectionStackType>::New();

  auto constantVolumeSeriesSource = rtk::ConstantImageSource<VolumeSeriesType>::New();

  // Volume metadata
  auto fourDOrigin = itk::MakePoint(-127., -127., -127., 0.);
#if FAST_TESTS_NO_CHECKS
  auto fourDSize = itk::MakeSize(2, 2, 2, 2);
  auto fourDSpacing = itk::MakeVector(252., 252., 252., 1.);
#else
  auto fourDSize = itk::MakeSize(64, 64, 64, 5);
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

  // Projections metadata
  auto origin = itk::MakePoint(-255., -255., -255.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(2, 2, NumberOfProjectionImages);
  auto spacing = itk::MakeVector(504., 504., 504.);
#else
  auto size = itk::MakeSize(64, 64, NumberOfProjectionImages);
  auto spacing = itk::MakeVector(8., 8., 8.);
#endif
  randomProjectionStackSource->SetOrigin(origin);
  randomProjectionStackSource->SetSpacing(spacing);
  randomProjectionStackSource->SetSize(size);
  randomProjectionStackSource->SetMin(0.);
  randomProjectionStackSource->SetMax(1.);

  constantProjectionStackSource->SetOrigin(origin);
  constantProjectionStackSource->SetSpacing(spacing);
  constantProjectionStackSource->SetSize(size);
  constantProjectionStackSource->SetConstant(0.);

  // Update all sources
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomVolumeSeriesSource->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantVolumeSeriesSource->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomProjectionStackSource->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantProjectionStackSource->Update());

  // Geometry object
  auto geometry = rtk::ThreeDCircularProjectionGeometry::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);

  // Read the phases file
  auto phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(argv[1]);
  phaseReader->SetNumberOfReconstructedFrames(fourDSize[3]);
  phaseReader->Update();

  std::cout << "\n\n****** 4D to projection stack ******" << std::endl;

  auto jfw = rtk::JosephForwardProjectionImageFilter<ProjectionStackType, ProjectionStackType>::New();

  auto fw = rtk::FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>::New();
  fw->SetInputProjectionStack(constantProjectionStackSource->GetOutput());
  fw->SetInputVolumeSeries(randomVolumeSeriesSource->GetOutput());
  fw->SetForwardProjectionFilter(jfw.GetPointer());
  fw->SetGeometry(geometry);
  fw->SetWeights(phaseReader->GetOutput());
  fw->SetSignal(rtk::ReadSignalFile(argv[1]));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(fw->Update());

  std::cout << "\n\n****** Projection stack to 4D ******" << std::endl;

  auto jbp = rtk::JosephBackProjectionImageFilter<ProjectionStackType, ProjectionStackType>::New();

  auto bp = rtk::ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>::New();
  bp->SetInputVolumeSeries(constantVolumeSeriesSource->GetOutput());
  bp->SetInputProjectionStack(randomProjectionStackSource->GetOutput());
  bp->SetBackProjectionFilter(jbp.GetPointer());
  bp->SetGeometry(geometry.GetPointer());
  bp->SetWeights(phaseReader->GetOutput());
  bp->SetSignal(rtk::ReadSignalFile(argv[1]));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(bp->Update());

  CheckScalarProducts<VolumeSeriesType, ProjectionStackType>(
    randomVolumeSeriesSource->GetOutput(), bp->GetOutput(), randomProjectionStackSource->GetOutput(), fw->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
