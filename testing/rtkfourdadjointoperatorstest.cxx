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

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > ProjectionStackType;
  typedef itk::CudaImage< OutputPixelType, Dimension + 1 > VolumeSeriesType;
#else
  typedef itk::Image< OutputPixelType, Dimension > ProjectionStackType;
  typedef itk::Image< OutputPixelType, Dimension + 1 > VolumeSeriesType;
#endif

#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfProjectionImages = 3;
#else
  const unsigned int NumberOfProjectionImages = 64;
#endif


  // Random image sources
  typedef itk::RandomImageSource< ProjectionStackType > RandomProjectionStackSourceType;
  RandomProjectionStackSourceType::Pointer randomProjectionStackSource  = RandomProjectionStackSourceType::New();

  typedef itk::RandomImageSource< VolumeSeriesType > RandomVolumeSeriesSourceType;
  RandomVolumeSeriesSourceType::Pointer randomVolumeSeriesSource  = RandomVolumeSeriesSourceType::New();

  // Constant sources
  typedef rtk::ConstantImageSource< ProjectionStackType > ConstantProjectionStackSourceType;
  ConstantProjectionStackSourceType::Pointer constantProjectionStackSource  = ConstantProjectionStackSourceType::New();

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
  fourDSize[2] = 2;
  fourDSize[3] = 2;
  fourDSpacing[0] = 252.;
  fourDSpacing[1] = 252.;
  fourDSpacing[2] = 252.;
  fourDSpacing[3] = 1.;
#else
  fourDSize[0] = 64;
  fourDSize[1] = 64;
  fourDSize[2] = 64;
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

  // Projections metadata
  ProjectionStackType::PointType origin;
  ProjectionStackType::SizeType size;
  ProjectionStackType::SpacingType spacing;

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
  randomProjectionStackSource->SetOrigin( origin );
  randomProjectionStackSource->SetSpacing( spacing );
  randomProjectionStackSource->SetSize( size );
  randomProjectionStackSource->SetMin( 0. );
  randomProjectionStackSource->SetMax( 1. );

  constantProjectionStackSource->SetOrigin( origin );
  constantProjectionStackSource->SetSpacing( spacing );
  constantProjectionStackSource->SetSize( size );
  constantProjectionStackSource->SetConstant( 0. );

  // Update all sources
  TRY_AND_EXIT_ON_ITK_EXCEPTION( randomVolumeSeriesSource->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( constantVolumeSeriesSource->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( randomProjectionStackSource->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( constantProjectionStackSource->Update() );

  // Geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages);

  // Read the phases file
  rtk::PhasesToInterpolationWeights::Pointer phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(std::string(RTK_DATA_ROOT) +
                           std::string("/Input/Phases/phases_slow.txt"));
  phaseReader->SetNumberOfReconstructedFrames( fourDSize[3] );
  phaseReader->Update();

  std::cout << "\n\n****** 4D to projection stack ******" << std::endl;

  typedef rtk::JosephForwardProjectionImageFilter<ProjectionStackType, ProjectionStackType> JosephForwardProjectorType;
  JosephForwardProjectorType::Pointer jfw = JosephForwardProjectorType::New();

  typedef rtk::FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType> FourDToProjectionStackFilterType;
  FourDToProjectionStackFilterType::Pointer fw = FourDToProjectionStackFilterType::New();
  fw->SetInputProjectionStack(constantProjectionStackSource->GetOutput());
  fw->SetInputVolumeSeries(randomVolumeSeriesSource->GetOutput());
  fw->SetForwardProjectionFilter( jfw.GetPointer() );
  fw->SetGeometry( geometry );
  fw->SetWeights(phaseReader->GetOutput());
  fw->SetSignal(rtk::ReadSignalFile(std::string(RTK_DATA_ROOT) +
                           std::string("/Input/Phases/phases_slow.txt")));
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fw->Update() );

  std::cout << "\n\n****** Projection stack to 4D ******" << std::endl;

  typedef rtk::JosephBackProjectionImageFilter<ProjectionStackType, ProjectionStackType> JosephBackProjectorType;
  JosephBackProjectorType::Pointer jbp = JosephBackProjectorType::New();

  typedef rtk::ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType> ProjectionStackToFourDFilterType;
  ProjectionStackToFourDFilterType::Pointer bp = ProjectionStackToFourDFilterType::New();
  bp->SetInputVolumeSeries(constantVolumeSeriesSource->GetOutput());
  bp->SetInputProjectionStack(randomProjectionStackSource->GetOutput());
  bp->SetBackProjectionFilter( jbp.GetPointer() );
  bp->SetGeometry( geometry.GetPointer() );
  bp->SetWeights(phaseReader->GetOutput());
  bp->SetSignal(rtk::ReadSignalFile(std::string(RTK_DATA_ROOT) +
                           std::string("/Input/Phases/phases_slow.txt")));
  TRY_AND_EXIT_ON_ITK_EXCEPTION( bp->Update() );

  CheckScalarProducts<VolumeSeriesType, ProjectionStackType>(randomVolumeSeriesSource->GetOutput(), bp->GetOutput(), randomProjectionStackSource->GetOutput(), fw->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
