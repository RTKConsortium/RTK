#include <itkPasteImageFilter.h>
#include <itksys/SystemTools.hxx>
#include <itkJoinSeriesImageFilter.h>

#include "rtkTest.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"
#include "rtkCyclicDeformationImageFilter.h"
#include "rtkFourDROOSTERConeBeamReconstructionFilter.h"
#include "rtkPhasesToInterpolationWeights.h"

/**
 * \file rtkfourdroostertest.cxx
 *
 * \brief Functional test for classes performing 4D ROOSTER
 * reconstruction.
 *
 * This test generates the projections of a phantom, which consists of two
 * ellipsoids (one of them moving). The resulting moving phantom is
 * reconstructed using the 4D ROOSTER algorithm and the generated
 * result is compared to the expected results (analytical computation).
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
  auto tomographySource = ConstantImageSourceType::New();
  auto origin = itk::MakePoint(-63., -31., -63.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(8, 8, 8);
  auto spacing = itk::MakeVector(16., 8., 16.);
#else
  auto size = itk::MakeSize(32, 16, 32);
  auto spacing = itk::MakeVector(4., 4., 4.);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  auto fourdSource = rtk::ConstantImageSource<VolumeSeriesType>::New();
  auto fourDOrigin = itk::MakePoint(-63., -31., -63., 0.);
#if FAST_TESTS_NO_CHECKS
  auto fourDSize = itk::MakeSize(8, 8, 8, 2);
  auto fourDSpacing = itk::MakeVector(16., 8., 16., 1.);
#else
  auto fourDSize = itk::MakeSize(32, 16, 32, 8);
  auto fourDSpacing = itk::MakeVector(4., 4., 4., 1.);
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
  size = itk::MakeSize(64, 64, NumberOfProjectionImages);
  spacing = itk::MakeVector(8., 8., 1.);
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
  auto destinationIndex = itk::MakeIndex(0, 0, 0);
  auto pasteFilter = itk::PasteImageFilter<ProjectionStackType, ProjectionStackType, ProjectionStackType>::New();
  pasteFilter->SetDestinationImage(projectionsSource->GetOutput());

#ifdef USE_CUDA
  std::string signalFileName = "signal_4DRooster_cuda.txt";
#else
  std::string signalFileName = "signal_4DRooster.txt";
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
    e1->SetDensity(2.);
    e1->SetAxis(semiprincipalaxis);
    e1->SetCenter(center);
    e1->SetAngle(0.);
    e1->InPlaceOff();
    e1->Update();

    // Ellipse 2
    auto e2 = REIType::New();
    semiprincipalaxis.Fill(8.);
    center[0] = 4 * (itk::Math::abs((4 + noProj) % 8 - 4.) - 2.);
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
  signalFile.close();

  // Create a vector field and its (very rough) inverse
  using IteratorType = itk::ImageRegionIteratorWithIndex<DVFSequenceImageType>;

  auto deformationField = DVFSequenceImageType::New();
  auto inverseDeformationField = DVFSequenceImageType::New();

  auto                             sizeMotion = itk::MakeSize(fourDSize[0], fourDSize[1], fourDSize[2], 2);
  auto                             originMotion = itk::MakePoint(-63., -31., -63., 0.);
  DVFSequenceImageType::RegionType regionMotion;
  regionMotion.SetSize(sizeMotion);

  auto spacingMotion = itk::MakeVector(fourDSpacing[0], fourDSpacing[1], fourDSpacing[2], fourDSpacing[3]);
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

    if (0.3 * toCenter[0] * toCenter[0] + toCenter[1] * toCenter[1] + toCenter[2] * toCenter[2] < 40)
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

  // Ground truth
  auto * Volumes = new VolumeType::Pointer[fourDSize[3]];
  auto   join = itk::JoinSeriesImageFilter<VolumeType, VolumeSeriesType>::New();

  for (itk::SizeValueType n = 0; n < fourDSize[3]; n++)
  {
    // Ellipse 1
    using DEType = rtk::DrawEllipsoidImageFilter<VolumeType, VolumeType>;
    auto de1 = DEType::New();
    de1->SetInput(tomographySource->GetOutput());
    de1->SetDensity(2.);
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

  // ROI
  using DEType = rtk::DrawEllipsoidImageFilter<VolumeType, VolumeType>;
  auto roi = DEType::New();
  roi->SetInput(tomographySource->GetOutput());
  roi->SetDensity(1.);
  DEType::VectorType axis;
  axis.Fill(15.);
  axis[0] = 20;
  roi->SetAxis(axis);
  DEType::VectorType center;
  center.Fill(0.);
  roi->SetCenter(center);
  roi->SetAngle(0.);
  roi->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(roi->Update())

  // Read the phases file
  auto phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(signalFileName);
  phaseReader->SetNumberOfReconstructedFrames(fourDSize[3]);
  phaseReader->Update();

  // Set the forward and back projection filters to be used
  using ROOSTERFilterType = rtk::FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>;
  auto rooster = ROOSTERFilterType::New();
  rooster->SetInputVolumeSeries(fourdSource->GetOutput());
  rooster->SetInputProjectionStack(pasteFilter->GetOutput());
  rooster->SetGeometry(geometry);
  rooster->SetWeights(phaseReader->GetOutput());
  rooster->SetSignal(rtk::ReadSignalFile(signalFileName));
  rooster->SetGeometry(geometry);
  rooster->SetCG_iterations(2);
  rooster->SetMainLoop_iterations(2);

  rooster->SetTV_iterations(3);
  rooster->SetGammaTVSpace(1);
  rooster->SetGammaTVTime(0.1);

  rooster->SetSoftThresholdWavelets(0.1);
  rooster->SetOrder(3);
  rooster->SetNumberOfLevels(3);

  rooster->SetLambdaL0Time(0.1);
  rooster->SetL0_iterations(5);

  std::cout << "\n\n****** Case 1: Joseph forward projector, voxel-based back projector, positivity, motion mask, "
               "wavelets spatial denoising, TV temporal denoising, no warping ******"
            << std::endl;

  rooster->SetBackProjectionFilter(ROOSTERFilterType::BP_VOXELBASED);
  rooster->SetForwardProjectionFilter(ROOSTERFilterType::FP_JOSEPH);

  rooster->SetPerformPositivity(true);
  rooster->SetPerformMotionMask(true);
  rooster->SetMotionMask(roi->GetOutput());
  rooster->SetPerformTVSpatialDenoising(false);
  rooster->SetPerformWaveletsSpatialDenoising(true);
  rooster->SetPerformTVTemporalDenoising(true);
  rooster->SetPerformL0TemporalDenoising(false);
  rooster->SetPerformWarping(false);
  rooster->SetComputeInverseWarpingByConjugateGradient(false);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(rooster->Update());

  CheckImageQuality<VolumeSeriesType>(rooster->GetOutput(), join->GetOutput(), 0.25, 15, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Joseph forward projector, voxel-based back projector, positivity, no motion mask, "
               "TV spatial denoising, L0 temporal denoising, motion compensation (nearest neighbor interpolation). "
               "Inverse warping by conjugate gradient ******"
            << std::endl;

  rooster->SetBackProjectionFilter(ROOSTERFilterType::BP_VOXELBASED);
  rooster->SetForwardProjectionFilter(ROOSTERFilterType::FP_JOSEPH);

  rooster->SetPerformPositivity(true);
  rooster->SetPerformMotionMask(false);
  rooster->SetPerformTVSpatialDenoising(true);
  rooster->SetPerformWaveletsSpatialDenoising(false);
  rooster->SetPerformTVTemporalDenoising(false);
  rooster->SetPerformL0TemporalDenoising(true);
  rooster->SetPerformWarping(true);
  rooster->SetDisplacementField(deformationField);
  rooster->SetComputeInverseWarpingByConjugateGradient(true);
  rooster->SetUseNearestNeighborInterpolationInWarping(true);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(rooster->Update());

  CheckImageQuality<VolumeSeriesType>(rooster->GetOutput(), join->GetOutput(), 0.25, 15, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 3: Joseph forward projector, voxel-based back projector, no positivity, motion mask, "
               "no spatial denoising, motion compensation and temporal TV denoising. Inverse warping by warping with "
               "approximate inverse DVF ******"
            << std::endl;

  rooster->SetBackProjectionFilter(ROOSTERFilterType::BP_VOXELBASED);
  rooster->SetForwardProjectionFilter(ROOSTERFilterType::FP_JOSEPH);

  rooster->SetPerformPositivity(false);
  rooster->SetPerformMotionMask(true);
  rooster->SetMotionMask(roi->GetOutput());
  rooster->SetPerformTVSpatialDenoising(false);
  rooster->SetPerformWaveletsSpatialDenoising(false);
  rooster->SetPerformTVTemporalDenoising(true);
  rooster->SetPerformL0TemporalDenoising(false);
  rooster->SetPerformWarping(true);
  rooster->SetDisplacementField(deformationField);
  rooster->SetComputeInverseWarpingByConjugateGradient(false);
  rooster->SetInverseDisplacementField(inverseDeformationField);
  rooster->SetUseNearestNeighborInterpolationInWarping(false);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(rooster->Update());

  CheckImageQuality<VolumeSeriesType>(rooster->GetOutput(), join->GetOutput(), 0.25, 15, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 4: CUDA forward and back projectors, only L0 temporal denoising ******" << std::endl;

  rooster->SetBackProjectionFilter(ROOSTERFilterType::BP_CUDAVOXELBASED); // Cuda voxel based
  rooster->SetForwardProjectionFilter(ROOSTERFilterType::FP_CUDARAYCAST); // Cuda ray cast

  rooster->SetPerformPositivity(false);
  rooster->SetPerformMotionMask(false);
  rooster->SetPerformTVSpatialDenoising(false);
  rooster->SetPerformWaveletsSpatialDenoising(false);
  rooster->SetPerformTVTemporalDenoising(false);
  rooster->SetPerformL0TemporalDenoising(true);
  rooster->SetPerformWarping(false);
  rooster->SetComputeInverseWarpingByConjugateGradient(false);
  rooster->SetUseNearestNeighborInterpolationInWarping(false);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(rooster->Update());

  CheckImageQuality<VolumeSeriesType>(rooster->GetOutput(), join->GetOutput(), 0.25, 15, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  itksys::SystemTools::RemoveFile(signalFileName.c_str());
  delete[] Volumes;

  return EXIT_SUCCESS;
}
