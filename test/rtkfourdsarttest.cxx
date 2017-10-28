#include <itkImageRegionConstIterator.h>
#include <itkPasteImageFilter.h>
#include <itksys/SystemTools.hxx>
#include <itkJoinSeriesImageFilter.h>

#include "rtkTest.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"
#include "rtkCyclicDeformationImageFilter.h"
#include "rtkFourDSARTConeBeamReconstructionFilter.h"
#include "rtkPhasesToInterpolationWeights.h"

/**
 * \file rtkfourdsarttest.cxx
 *
 * \brief Functional test for classes performing 4D SART
 * reconstruction.
 *
 * This test generates the projections of a phantom, which consists of two
 * ellipsoids (one of them moving). The resulting moving phantom is
 * reconstructed using 4D SART and the generated
 * result is compared to the expected results (analytical computation).
 *
 * \author Cyril Mory
 */

int main(int, char** )
{
  typedef float                             OutputPixelType;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, 4 >  VolumeSeriesType;
  typedef itk::CudaImage< OutputPixelType, 3 >  ProjectionStackType;
  typedef itk::CudaImage< OutputPixelType, 3 >  VolumeType;
#else
  typedef itk::Image< OutputPixelType, 4 >  VolumeSeriesType;
  typedef itk::Image< OutputPixelType, 3 >  ProjectionStackType;
  typedef itk::Image< OutputPixelType, 3 >  VolumeType;
#endif

#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfProjectionImages = 5;
#else
  const unsigned int NumberOfProjectionImages = 64;
#endif

  // Constant image sources
  typedef rtk::ConstantImageSource< VolumeType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  typedef rtk::ConstantImageSource< VolumeSeriesType > FourDSourceType;
  FourDSourceType::PointType fourDOrigin;
  FourDSourceType::SizeType fourDSize;
  FourDSourceType::SpacingType fourDSpacing;

  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  origin[0] = -63.;
  origin[1] = -31.;
  origin[2] = -63.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 8;
  size[1] = 8;
  size[2] = 8;
  spacing[0] = 16.;
  spacing[1] = 8.;
  spacing[2] = 16.;
#else
  size[0] = 32;
  size[1] = 16;
  size[2] = 32;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
#endif
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  FourDSourceType::Pointer fourdSource  = FourDSourceType::New();
  fourDOrigin[0] = -63.;
  fourDOrigin[1] = -31.;
  fourDOrigin[2] = -63.;
  fourDOrigin[3] = 0;
#if FAST_TESTS_NO_CHECKS
  fourDSize[0] = 8;
  fourDSize[1] = 8;
  fourDSize[2] = 8;
  fourDSize[3] = 2;
  fourDSpacing[0] = 16.;
  fourDSpacing[1] = 8.;
  fourDSpacing[2] = 16.;
  fourDSpacing[3] = 1.;
#else
  fourDSize[0] = 32;
  fourDSize[1] = 16;
  fourDSize[2] = 32;
  fourDSize[3] = 8;
  fourDSpacing[0] = 4.;
  fourDSpacing[1] = 4.;
  fourDSpacing[2] = 4.;
  fourDSpacing[3] = 1.;
#endif
  fourdSource->SetOrigin( fourDOrigin );
  fourdSource->SetSpacing( fourDSpacing );
  fourdSource->SetSize( fourDSize );
  fourdSource->SetConstant( 0. );

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -254.;
  origin[1] = -254.;
  origin[2] = -254.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 32;
  size[1] = 32;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 32.;
  spacing[1] = 32.;
  spacing[2] = 32.;
#else
  size[0] = 64;
  size[1] = 64;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 1.;
#endif
  projectionsSource->SetOrigin( origin );
  projectionsSource->SetSpacing( spacing );
  projectionsSource->SetSize( size );
  projectionsSource->SetConstant( 0. );

  ConstantImageSourceType::Pointer oneProjectionSource = ConstantImageSourceType::New();
  size[2] = 1;
  oneProjectionSource->SetOrigin( origin );
  oneProjectionSource->SetSpacing( spacing );
  oneProjectionSource->SetSize( size );
  oneProjectionSource->SetConstant( 0. );

  // Geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();

  // Projections
  typedef rtk::RayEllipsoidIntersectionImageFilter<VolumeType, ProjectionStackType> REIType;
  typedef itk::PasteImageFilter <ProjectionStackType, ProjectionStackType, ProjectionStackType > PasteImageFilterType;
  ProjectionStackType::IndexType destinationIndex;
  destinationIndex[0] = 0;
  destinationIndex[1] = 0;
  destinationIndex[2] = 0;
  PasteImageFilterType::Pointer pasteFilter = PasteImageFilterType::New();
  pasteFilter->SetDestinationImage(projectionsSource->GetOutput());

  std::ofstream signalFile("signal.txt");
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    {
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

    // Geometry object
    GeometryType::Pointer oneProjGeometry = GeometryType::New();
    oneProjGeometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

    // Ellipse 1
    REIType::Pointer e1 = REIType::New();
    REIType::VectorType semiprincipalaxis, center;
    semiprincipalaxis.Fill(60.);
    semiprincipalaxis[1]=30;
    center.Fill(0.);
    e1->SetInput(oneProjectionSource->GetOutput());
    e1->SetGeometry(oneProjGeometry);
    e1->SetDensity(2.);
    e1->SetAxis(semiprincipalaxis);
    e1->SetCenter(center);
    e1->SetAngle(0.);
    e1->InPlaceOff();
    e1->Update();

    // Ellipse 2
    REIType::Pointer e2 = REIType::New();
    semiprincipalaxis.Fill(8.);
    center[0] = 4*(vcl_abs( (4+noProj) % 8 - 4.) - 2.);
    center[1] = 0.;
    center[2] = 0.;
    e2->SetInput(e1->GetOutput());
    e2->SetGeometry(oneProjGeometry);
    e2->SetDensity(-1.);
    e2->SetAxis(semiprincipalaxis);
    e2->SetCenter(center);
    e2->SetAngle(0.);
    e2->Update();

    // Adding each projection to the projection stack
    if (noProj > 0) // After the first projection, we use the output as input
      {
      ProjectionStackType::Pointer wholeImage = pasteFilter->GetOutput();
      wholeImage->DisconnectPipeline();
      pasteFilter->SetDestinationImage(wholeImage);
      }
    pasteFilter->SetSourceImage(e2->GetOutput());
    pasteFilter->SetSourceRegion(e2->GetOutput()->GetLargestPossibleRegion());
    pasteFilter->SetDestinationIndex(destinationIndex);
    pasteFilter->UpdateLargestPossibleRegion();
    destinationIndex[2]++;

    // Signal
    signalFile << (noProj % 8) / 8. << std::endl;
    }

//  pasteFilter->GetOutput()->Print(std::cout);

  // Ground truth
  VolumeType::Pointer * Volumes = new VolumeType::Pointer[fourDSize[3]];
  typedef itk::JoinSeriesImageFilter<VolumeType, VolumeSeriesType> JoinFilterType;
  JoinFilterType::Pointer join = JoinFilterType::New();

  for (itk::SizeValueType n = 0; n < fourDSize[3]; n++)
    {
    // Ellipse 1
    typedef rtk::DrawEllipsoidImageFilter<VolumeType, VolumeType> DEType;
    DEType::Pointer de1 = DEType::New();
    de1->SetInput( tomographySource->GetOutput() );
    de1->SetDensity(2.);
    DEType::VectorType axis;
    axis.Fill(60.);
    axis[1]=30;
    de1->SetAxis(axis);
    DEType::VectorType center;
    center.Fill(0.);
    de1->SetCenter(center);
    de1->SetAngle(0.);
    de1->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION( de1->Update() )

    // Ellipse 2
    DEType::Pointer de2 = DEType::New();
    de2->SetInput(de1->GetOutput());
    de2->SetDensity(-1.);
    DEType::VectorType axis2;
    axis2.Fill(8.);
    de2->SetAxis(axis2);
    DEType::VectorType center2;
    center2[0] = 4*(vcl_abs( (4+n) % 8 - 4.) - 2.);
    center2[1] = 0.;
    center2[2] = 0.;
    de2->SetCenter(center2);
    de2->SetAngle(0.);
    de2->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION( de2->Update() );

    Volumes[n] = de2->GetOutput();
    Volumes[n]->DisconnectPipeline();
    join->SetInput(n, Volumes[n]);
    }
  join->Update();

  // Read the phases file
  rtk::PhasesToInterpolationWeights::Pointer phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName("signal.txt");
  phaseReader->SetNumberOfReconstructedFrames( fourDSize[3] );
  phaseReader->Update();

  // Set the forward and back projection filters to be used
  typedef rtk::FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType> FourDSARTFilterType;
  FourDSARTFilterType::Pointer fourdsart = FourDSARTFilterType::New();
  fourdsart->SetInputVolumeSeries(fourdSource->GetOutput() );
  fourdsart->SetInputProjectionStack(pasteFilter->GetOutput());
  fourdsart->SetGeometry(geometry);
  fourdsart->SetNumberOfIterations(3);
  fourdsart->SetWeights(phaseReader->GetOutput());
  fourdsart->SetSignal(rtk::ReadSignalFile("signal.txt"));
  fourdsart->SetEnforcePositivity(true);

  std::cout << "\n\n****** Case 1: Joseph forward projector, Voxel-Based back projector, CPU interpolation and splat ******" << std::endl;

  fourdsart->SetBackProjectionFilter( 0 ); // Voxel based
  fourdsart->SetForwardProjectionFilter( 0 ); // Joseph
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fourdsart->Update() );

  CheckImageQuality<VolumeSeriesType>(fourdsart->GetOutput(), join->GetOutput(), 0.22, 12, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Joseph forward projector, Voxel-Based back projector, CPU interpolation and splat, 2 projections per subset ******" << std::endl;

  fourdsart->SetBackProjectionFilter( 0 ); // Voxel based
  fourdsart->SetForwardProjectionFilter( 0 ); // Joseph
  fourdsart->SetNumberOfProjectionsPerSubset(2);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fourdsart->Update() );

  CheckImageQuality<VolumeSeriesType>(fourdsart->GetOutput(), join->GetOutput(), 0.35, 12, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 3: CUDA ray cast forward projector, CUDA Voxel-Based back projector, GPU interpolation and splat ******" << std::endl;

  fourdsart->SetBackProjectionFilter( 2 ); // Cuda voxel based
  fourdsart->SetForwardProjectionFilter( 2 ); // Cuda ray cast
  fourdsart->SetNumberOfProjectionsPerSubset(1);
  fourdsart->SetNumberOfIterations(3);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fourdsart->Update() );

  CheckImageQuality<VolumeSeriesType>(fourdsart->GetOutput(), join->GetOutput(), 0.22, 12, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  itksys::SystemTools::RemoveFile("signal.txt");
  delete[] Volumes;

  return EXIT_SUCCESS;
}
