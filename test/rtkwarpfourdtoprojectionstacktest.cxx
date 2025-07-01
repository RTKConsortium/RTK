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

int
main(int, char **)
{
  using OutputPixelType = float;

  using DVFVectorType = itk::CovariantVector<OutputPixelType, 3>;

#ifdef USE_CUDA
  using VolumeSeriesType = itk::CudaImage<OutputPixelType, 4>;
  using ProjectionStackType = itk::CudaImage<OutputPixelType, 3>;
  using VolumeType = itk::CudaImage<OutputPixelType, 3>;
  using DVFSequenceImageType = itk::CudaImage<DVFVectorType, VolumeSeriesType::ImageDimension>;
#else
  using VolumeSeriesType = itk::Image<OutputPixelType, 4>;
  using ProjectionStackType = itk::Image<OutputPixelType, 3>;
  using VolumeType = itk::Image<OutputPixelType, 3>;
  using DVFSequenceImageType = itk::Image<DVFVectorType, VolumeSeriesType::ImageDimension>;
#endif

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 5;
#else
  constexpr unsigned int NumberOfProjectionImages = 64;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<VolumeType>;
  using FourDSourceType = rtk::ConstantImageSource<VolumeSeriesType>;

  auto tomographySource = ConstantImageSourceType::New();
  auto origin = itk::MakePoint(-63.0, -31.0, -63.0);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(8, 8, 8);
  auto spacing = itk::MakeVector(16., 8., 16.);
#else
  auto size = itk::MakeSize(64, 64, 64);
  auto spacing = itk::MakeVector(2., 1., 2.);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  auto fourdSource = FourDSourceType::New();
  auto fourDOrigin = itk::MakePoint(-63.0, -31.5, -63.0, 0.0);
#if FAST_TESTS_NO_CHECKS
  auto fourDSize = itk::MakeSize(8, 8, 8, 2);
  auto fourDSpacing = itk::MakeVector(16., 8., 16., 1.);
#else
  auto fourDSize = itk::MakeSize(64, 64, 64, 8);
  auto fourDSpacing = itk::MakeVector(2., 1., 2., 1.);
#endif
  fourdSource->SetOrigin(fourDOrigin);
  fourdSource->SetSpacing(fourDSpacing);
  fourdSource->SetSize(fourDSize);
  fourdSource->SetConstant(0.);

  auto projectionsSource = ConstantImageSourceType::New();
  origin = itk::MakePoint(-254., -254., -254.);
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(32, 32, NumberOfProjectionImages);
  spacing = itk::MakeVector(32., 32., 32.);
#else
  size = itk::MakeSize(128, 128, NumberOfProjectionImages);
  spacing = itk::MakeVector(4., 4., 1.);
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  auto oneProjectionSource = ConstantImageSourceType::New();
  size[2] = 1;
  oneProjectionSource->SetOrigin(origin);
  oneProjectionSource->SetSpacing(spacing);
  oneProjectionSource->SetSize(size);
  oneProjectionSource->SetConstant(0.);

  // Geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto geometry = GeometryType::New();

  // Projections
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<VolumeType, ProjectionStackType>;
  using PasteImageFilterType = itk::PasteImageFilter<ProjectionStackType, ProjectionStackType, ProjectionStackType>;
  ProjectionStackType::IndexType destinationIndex;
  destinationIndex[0] = 0;
  destinationIndex[1] = 0;
  destinationIndex[2] = 0;

  auto pasteFilter = PasteImageFilterType::New();
  pasteFilter->SetDestinationImage(projectionsSource->GetOutput());

  auto pasteFilterStaticProjections = PasteImageFilterType::New();
  pasteFilterStaticProjections->SetDestinationImage(projectionsSource->GetOutput());

#ifdef USE_CUDA
  std::string signalFileName = "signal_bw_cuda.txt";
#else
  std::string signalFileName = "signal_bw.txt";
#endif

  std::ofstream signalFile(signalFileName.c_str());
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
  {
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

    // Geometry object
    auto oneProjGeometry = GeometryType::New();
    oneProjGeometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

    // Ellipse 1
    auto e1 = REIType::New();
    auto semiprincipalaxis = itk::MakeVector(60., 30., 60.);
    auto center = itk::MakeVector(0., 0., 0.);
    e1->SetInput(oneProjectionSource->GetOutput());
    e1->SetGeometry(oneProjGeometry);
    e1->SetDensity(1.);
    e1->SetAxis(semiprincipalaxis);
    e1->SetCenter(center);
    e1->SetAngle(0.);
    e1->InPlaceOff();
    e1->Update();

    // Ellipse 2
    auto e2 = REIType::New();
    semiprincipalaxis.Fill(8.);
    center[0] = 4 * (itk::Math::abs((4 + noProj) % 8 - 4.) - 2.);
    e2->SetInput(e1->GetOutput());
    e2->SetGeometry(oneProjGeometry);
    e2->SetDensity(-1.);
    e2->SetAxis(semiprincipalaxis);
    e2->SetCenter(center);
    e2->SetAngle(0.);
    e2->Update();

    // Ellipse 2 without motion
    auto e2static = REIType::New();
    center[0] = 0;
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
  signalFile.close();

  // Create a vector field and its (very rough) inverse
  using IteratorType = itk::ImageRegionIteratorWithIndex<DVFSequenceImageType>;

  auto deformationField = DVFSequenceImageType::New();
  auto inverseDeformationField = DVFSequenceImageType::New();

  auto sizeMotion = itk::MakeSize(fourDSize[0], fourDSize[1], fourDSize[2], 2);
  auto originMotion = itk::MakePoint(-63., -31., -63., 0.);
  auto spacingMotion = itk::MakeVector(fourDSpacing[0], fourDSpacing[1], fourDSpacing[2], fourDSpacing[3]);
  DVFSequenceImageType::RegionType regionMotion;
  regionMotion.SetSize(sizeMotion);

  deformationField->SetRegions(regionMotion);
  deformationField->SetOrigin(originMotion);
  deformationField->SetSpacing(spacingMotion);
  deformationField->Allocate();

  inverseDeformationField->SetRegions(regionMotion);
  inverseDeformationField->SetOrigin(originMotion);
  inverseDeformationField->SetSpacing(spacingMotion);
  inverseDeformationField->Allocate();

  // Vector Field initilization
  DVFVectorType vec;
  IteratorType  dvfIt(deformationField, deformationField->GetLargestPossibleRegion());
  IteratorType  idvfIt(inverseDeformationField, inverseDeformationField->GetLargestPossibleRegion());

  DVFSequenceImageType::OffsetType DVFCenter;
  DVFSequenceImageType::IndexType  toCenter;
  DVFCenter.Fill(0);
  DVFCenter[0] = sizeMotion[0] / 2;
  DVFCenter[1] = sizeMotion[1] / 2;
  DVFCenter[2] = sizeMotion[2] / 2;
  while (!dvfIt.IsAtEnd())
  {
    vec.Fill(0.);
    toCenter = dvfIt.GetIndex() - DVFCenter;

    if (0.3 * toCenter[0] * toCenter[0] + 0.5 * toCenter[1] * toCenter[1] + 0.5 * toCenter[2] * toCenter[2] < 40)
    {
      if (dvfIt.GetIndex()[3] == 0)
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
  auto * Volumes = new VolumeType::Pointer[fourDSize[3]];
  using JoinFilterType = itk::JoinSeriesImageFilter<VolumeType, VolumeSeriesType>;
  auto join = JoinFilterType::New();

  for (itk::SizeValueType n = 0; n < fourDSize[3]; n++)
  {
    // Ellipse 1
    using DEType = rtk::DrawEllipsoidImageFilter<VolumeType, VolumeType>;
    auto de1 = DEType::New();
    de1->SetInput(tomographySource->GetOutput());
    de1->SetDensity(1.);
    DEType::VectorType axis;
    axis.Fill(60.);
    axis[1] = 30;
    de1->SetAxis(axis);
    DEType::VectorType center;
    center.Fill(0.);
    de1->SetCenter(center);
    de1->SetAngle(0.);
    de1->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION(de1->Update())

    // Ellipse 2
    auto de2 = DEType::New();
    de2->SetInput(de1->GetOutput());
    de2->SetDensity(-1.);
    DEType::VectorType axis2;
    axis2.Fill(8.);
    de2->SetAxis(axis2);
    DEType::VectorType center2;
    center2[0] = 4 * (itk::Math::abs((4 + n) % 8 - 4.) - 2.);
    center2[1] = 0.;
    center2[2] = 0.;
    de2->SetCenter(center2);
    de2->SetAngle(0.);
    de2->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION(de2->Update());

    Volumes[n] = de2->GetOutput();
    Volumes[n]->DisconnectPipeline();
    join->SetInput(n, Volumes[n]);
  }
  join->Update();

  // Read the phases file
  auto phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(signalFileName);
  phaseReader->SetNumberOfReconstructedFrames(fourDSize[3]);
  phaseReader->Update();

  // Create and set the warped forward projection filter
  using WarpFourDToProjectionStackType = rtk::WarpFourDToProjectionStackImageFilter<VolumeSeriesType, VolumeType>;
  auto warpforwardproject = WarpFourDToProjectionStackType::New();
  warpforwardproject->SetInputVolumeSeries(join->GetOutput());
  warpforwardproject->SetInputProjectionStack(pasteFilter->GetOutput());
  warpforwardproject->SetGeometry(geometry);
  warpforwardproject->SetDisplacementField(deformationField);
  warpforwardproject->SetWeights(phaseReader->GetOutput());
  warpforwardproject->SetSignal(rtk::ReadSignalFile(signalFileName));

#ifndef USE_CUDA
  std::cout
    << "\n\n****** Case 1: Non-warped joseph forward projection (warped forward projection exists only in CUDA) ******"
    << std::endl;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(warpforwardproject->Update());

  // The warpforwardproject filter doesn't really need the data in pasteFilter->GetOutput().
  // During the update, its requested region is set to empty, and its buffered region follows.
  // To perform the CheckImageQuality, we need to recompute the data
  pasteFilter->UpdateLargestPossibleRegion();

  CheckImageQuality<ProjectionStackType>(warpforwardproject->GetOutput(), pasteFilter->GetOutput(), 0.25, 14, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 2: CUDA warped forward projection ******" << std::endl;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(warpforwardproject->Update());
  CheckImageQuality<ProjectionStackType>(
    warpforwardproject->GetOutput(), pasteFilterStaticProjections->GetOutput(), 0.25, 14, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  itksys::SystemTools::RemoveFile(signalFileName.c_str());
  delete[] Volumes;

  return EXIT_SUCCESS;
}
