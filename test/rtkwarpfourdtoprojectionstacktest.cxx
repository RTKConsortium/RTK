#include <itkPasteImageFilter.h>
#include <itksys/SystemTools.hxx>
#include <itkJoinSeriesImageFilter.h>

#include "rtkTest.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"
#include "rtkWarpFourDToProjectionStackImageFilter.h"
#include "rtkPhasesToInterpolationWeights.h"
#include "rtkGeneralPurposeFunctions.h"

/**
 * \file rtkwarpfourdtoprojectionstacktest.cxx
 *
 * \brief Functional test for classes performing motion-compensated
 * forward projection combining warping and forward projection
 *
 * This test generates the projections of a phantom, which consists of two
 * ellipsoids (one of them moving). The resulting moving phantom is
 * forward projected, each ray following a trajectory that matches the motion
 * The result is compared to the expected results (analytical computation).
 *
 * \author Cyril Mory
 */

int main(int, char** )
{
  typedef float                             OutputPixelType;

  typedef itk::CovariantVector< OutputPixelType, 3 > DVFVectorType;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, 4 >  VolumeSeriesType;
  typedef itk::CudaImage< OutputPixelType, 3 >  ProjectionStackType;
  typedef itk::CudaImage< OutputPixelType, 3 >  VolumeType;
  typedef itk::CudaImage<DVFVectorType, VolumeSeriesType::ImageDimension> DVFSequenceImageType;
  typedef itk::CudaImage<DVFVectorType, VolumeSeriesType::ImageDimension - 1> DVFImageType;
#else
  typedef itk::Image< OutputPixelType, 4 >  VolumeSeriesType;
  typedef itk::Image< OutputPixelType, 3 >  ProjectionStackType;
  typedef itk::Image< OutputPixelType, 3 >  VolumeType;
  typedef itk::Image<DVFVectorType, VolumeSeriesType::ImageDimension> DVFSequenceImageType;
  typedef itk::Image<DVFVectorType, VolumeSeriesType::ImageDimension - 1> DVFImageType;
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
  size[0] = 64;
  size[1] = 64;
  size[2] = 64;
  spacing[0] = 2.;
  spacing[1] = 1.;
  spacing[2] = 2.;
#endif
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  FourDSourceType::Pointer fourdSource  = FourDSourceType::New();
  fourDOrigin[0] = -63.;
  fourDOrigin[1] = -31.5;
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
  fourDSize[0] = 64;
  fourDSize[1] = 64;
  fourDSize[2] = 64;
  fourDSize[3] = 8;
  fourDSpacing[0] = 2.;
  fourDSpacing[1] = 1.;
  fourDSpacing[2] = 2.;
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
  size[0] = 128;
  size[1] = 128;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 4.;
  spacing[1] = 4.;
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

  PasteImageFilterType::Pointer pasteFilterStaticProjections = PasteImageFilterType::New();
  pasteFilterStaticProjections->SetDestinationImage(projectionsSource->GetOutput());

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
    e1->SetDensity(1.);
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

    // Ellipse 2 without motion
    REIType::Pointer e2static = REIType::New();
    semiprincipalaxis.Fill(8.);
    center[0] = 0;
    center[1] = 0.;
    center[2] = 0.;
    e2static->SetInput(e1->GetOutput());
    e2static->SetGeometry(oneProjGeometry);
    e2static->SetDensity(-1.);
    e2static->SetAxis(semiprincipalaxis);
    e2static->SetCenter(center);
    e2static->SetAngle(0.);
    e2static->Update();

    // Adding each projection to the projection stacks
    if (noProj > 0) // After the first projection, we use the output as input
      {
      ProjectionStackType::Pointer wholeImage = pasteFilter->GetOutput();
      wholeImage->DisconnectPipeline();
      pasteFilter->SetDestinationImage(wholeImage);

      ProjectionStackType::Pointer wholeImageStatic = pasteFilterStaticProjections->GetOutput();
      wholeImageStatic->DisconnectPipeline();
      pasteFilterStaticProjections->SetDestinationImage(wholeImageStatic);
      }
    pasteFilter->SetSourceImage(e2->GetOutput());
    pasteFilter->SetSourceRegion(e2->GetOutput()->GetLargestPossibleRegion());
    pasteFilter->SetDestinationIndex(destinationIndex);
    pasteFilter->UpdateLargestPossibleRegion();

    pasteFilterStaticProjections->SetSourceImage(e2static->GetOutput());
    pasteFilterStaticProjections->SetSourceRegion(e2static->GetOutput()->GetLargestPossibleRegion());
    pasteFilterStaticProjections->SetDestinationIndex(destinationIndex);
    pasteFilterStaticProjections->UpdateLargestPossibleRegion();

    destinationIndex[2]++;

    // Signal
    signalFile << (noProj % 8) / 8. << std::endl;
    }

  // Create a vector field and its (very rough) inverse
  typedef itk::ImageRegionIteratorWithIndex< DVFSequenceImageType > IteratorType;

  DVFSequenceImageType::Pointer deformationField = DVFSequenceImageType::New();
  DVFSequenceImageType::Pointer inverseDeformationField = DVFSequenceImageType::New();

  DVFSequenceImageType::IndexType startMotion;
  startMotion[0] = 0; // first index on X
  startMotion[1] = 0; // first index on Y
  startMotion[2] = 0; // first index on Z
  startMotion[3] = 0; // first index on t
  DVFSequenceImageType::SizeType sizeMotion;
  sizeMotion[0] = fourDSize[0];
  sizeMotion[1] = fourDSize[1];
  sizeMotion[2] = fourDSize[2];
  sizeMotion[3] = 2;
  DVFSequenceImageType::PointType originMotion;
  originMotion[0] = -63.;
  originMotion[1] = -31.;
  originMotion[2] = -63.;
  originMotion[3] = 0.;
  DVFSequenceImageType::RegionType regionMotion;
  regionMotion.SetSize( sizeMotion );
  regionMotion.SetIndex( startMotion );

  DVFSequenceImageType::SpacingType spacingMotion;
  spacingMotion[0] = fourDSpacing[0];
  spacingMotion[1] = fourDSpacing[1];
  spacingMotion[2] = fourDSpacing[2];
  spacingMotion[3] = fourDSpacing[3];

  deformationField->SetRegions( regionMotion );
  deformationField->SetOrigin(originMotion);
  deformationField->SetSpacing(spacingMotion);
  deformationField->Allocate();

  inverseDeformationField->SetRegions( regionMotion );
  inverseDeformationField->SetOrigin(originMotion);
  inverseDeformationField->SetSpacing(spacingMotion);
  inverseDeformationField->Allocate();

  // Vector Field initilization
  DVFVectorType vec;
  IteratorType dvfIt( deformationField, deformationField->GetLargestPossibleRegion() );
  IteratorType idvfIt( inverseDeformationField, inverseDeformationField->GetLargestPossibleRegion() );

  DVFSequenceImageType::OffsetType DVFCenter;
  DVFSequenceImageType::IndexType toCenter;
  DVFCenter.Fill(0);
  DVFCenter[0] = sizeMotion[0]/2;
  DVFCenter[1] = sizeMotion[1]/2;
  DVFCenter[2] = sizeMotion[2]/2;
  while (!dvfIt.IsAtEnd())
    {
    vec.Fill(0.);
    toCenter = dvfIt.GetIndex() - DVFCenter;

    if (0.3 * toCenter[0] * toCenter[0] + 0.5*toCenter[1] * toCenter[1] + 0.5*toCenter[2] * toCenter[2] < 40)
      {
      if(dvfIt.GetIndex()[3]==0)
        vec[0] = -8.;
      else
        vec[0] = 8.;
      }
    dvfIt.Set(vec);
    idvfIt.Set(-vec);

    ++dvfIt;
    ++idvfIt;
    }

  // Input 4D volume sequence
  VolumeType::Pointer * Volumes = new VolumeType::Pointer[fourDSize[3]];
  typedef itk::JoinSeriesImageFilter<VolumeType, VolumeSeriesType> JoinFilterType;
  JoinFilterType::Pointer join = JoinFilterType::New();

  for (itk::SizeValueType n = 0; n < fourDSize[3]; n++)
    {
    // Ellipse 1
    typedef rtk::DrawEllipsoidImageFilter<VolumeType, VolumeType> DEType;
    DEType::Pointer de1 = DEType::New();
    de1->SetInput( tomographySource->GetOutput() );
    de1->SetDensity(1.);
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

  // Create and set the warped forward projection filter
  typedef rtk::WarpFourDToProjectionStackImageFilter<VolumeSeriesType, VolumeType> WarpFourDToProjectionStackType;
  WarpFourDToProjectionStackType::Pointer warpforwardproject = WarpFourDToProjectionStackType::New();
  warpforwardproject->SetInputVolumeSeries(join->GetOutput() );
  warpforwardproject->SetInputProjectionStack(pasteFilter->GetOutput());
  warpforwardproject->SetGeometry(geometry);
  warpforwardproject->SetDisplacementField(deformationField);
  warpforwardproject->SetWeights(phaseReader->GetOutput());
  warpforwardproject->SetSignal(rtk::ReadSignalFile("signal.txt"));

#ifndef RTK_USE_CUDA
  std::cout << "\n\n****** Case 1: Non-warped joseph forward projection (warped forward projection exists only in CUDA) ******" << std::endl;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( warpforwardproject->Update() );

  // The warpforwardproject filter doesn't really need the data in pasteFilter->GetOutput().
  // During the update, its requested region is set to empty, and its buffered region follows.
  // To perform the CheckImageQuality, we need to recompute the data
  pasteFilter->UpdateLargestPossibleRegion();

  CheckImageQuality<ProjectionStackType>(warpforwardproject->GetOutput(), pasteFilter->GetOutput(), 0.25, 14, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 2: CUDA warped forward projection ******" << std::endl;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( warpforwardproject->Update() );
  CheckImageQuality<ProjectionStackType>(warpforwardproject->GetOutput(), pasteFilterStaticProjections->GetOutput(), 0.25, 14, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  itksys::SystemTools::RemoveFile("signal.txt");
  delete[] Volumes;

  return EXIT_SUCCESS;
}
