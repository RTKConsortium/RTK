#include <itksys/SystemTools.hxx>

#include "rtkTest.h"
#include "rtkFourDROOSTERConeBeamReconstructionFilter.h"
#include "rtkPhasesToInterpolationWeights.h"
#include "rtkFourDTestHelper.h"

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

  auto data = rtk::GenerateFourDTestData<OutputPixelType>(FAST_TESTS_NO_CHECKS);

  // ROI
  using DEType = rtk::DrawEllipsoidImageFilter<VolumeType, VolumeType>;
  auto roi = DEType::New();
  roi->SetInput(data.SingleVolume);
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
  phaseReader->SetFileName(data.SignalFileName);
  phaseReader->SetNumberOfReconstructedFrames(data.InitialVolumeSeries->GetLargestPossibleRegion().GetSize(3));
  phaseReader->Update();

  // Set the forward and back projection filters to be used
  using ROOSTERFilterType = rtk::FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>;
  auto rooster = ROOSTERFilterType::New();
  rooster->SetInputVolumeSeries(data.InitialVolumeSeries);
  rooster->SetInputProjectionStack(data.Projections);
  rooster->SetGeometry(data.Geometry);
  rooster->SetWeights(phaseReader->GetOutput());
  rooster->SetSignal(rtk::ReadSignalFile(data.SignalFileName));
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

  CheckImageQuality<VolumeSeriesType>(rooster->GetOutput(), data.GroundTruth, 0.25, 15, 2.0);
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
  rooster->SetDisplacementField(data.DVF);
  rooster->SetComputeInverseWarpingByConjugateGradient(true);
  rooster->SetUseNearestNeighborInterpolationInWarping(true);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(rooster->Update());

  CheckImageQuality<VolumeSeriesType>(rooster->GetOutput(), data.GroundTruth, 0.25, 15, 2.0);
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
  rooster->SetDisplacementField(data.DVF);
  rooster->SetComputeInverseWarpingByConjugateGradient(false);
  rooster->SetInverseDisplacementField(data.InverseDVF);
  rooster->SetUseNearestNeighborInterpolationInWarping(false);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(rooster->Update());

  CheckImageQuality<VolumeSeriesType>(rooster->GetOutput(), data.GroundTruth, 0.25, 15, 2.0);
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

  CheckImageQuality<VolumeSeriesType>(rooster->GetOutput(), data.GroundTruth, 0.25, 15, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  itksys::SystemTools::RemoveFile(data.SignalFileName.c_str());

  return EXIT_SUCCESS;
}
