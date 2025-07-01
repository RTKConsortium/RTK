/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtkfourdrooster_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkFourDROOSTERConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkSignalToInterpolationWeights.h"
#include "rtkReorderProjectionsImageFilter.h"

#ifdef RTK_USE_CUDA
#  include "itkCudaImage.h"
#endif
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkfourdrooster, args_info);

  using OutputPixelType = float;
  using DVFVectorType = itk::CovariantVector<OutputPixelType, 3>;

#ifdef RTK_USE_CUDA
  using VolumeSeriesType = itk::CudaImage<OutputPixelType, 4>;
  using ProjectionStackType = itk::CudaImage<OutputPixelType, 3>;
  using DVFSequenceImageType = itk::CudaImage<DVFVectorType, VolumeSeriesType::ImageDimension>;
#else
  using VolumeSeriesType = itk::Image<OutputPixelType, 4>;
  using ProjectionStackType = itk::Image<OutputPixelType, 3>;
  using DVFSequenceImageType = itk::Image<DVFVectorType, VolumeSeriesType::ImageDimension>;
#endif
  using VolumeType = ProjectionStackType;

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<ProjectionStackType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkfourdrooster>(reader, args_info);

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

  // Create input: either an existing volume read from a file or a blank image
  itk::ImageSource<VolumeSeriesType>::Pointer inputFilter;
  if (args_info.input_given)
  {
    // Read an existing image to initialize the volume
    using InputReaderType = itk::ImageFileReader<VolumeSeriesType>;
    auto inputReader = InputReaderType::New();
    inputReader->SetFileName(args_info.input_arg);
    inputFilter = inputReader;
  }
  else
  {
    // Create new empty volume
    using ConstantImageSourceType = rtk::ConstantImageSource<VolumeSeriesType>;
    auto constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkfourdrooster>(constantImageSource,
                                                                                           args_info);

    // GenGetOpt can't handle default arguments for multiple arguments like size or spacing.
    // The only default it accepts is to set all components of a multiple argument to the same value.
    // Default size is 256^4, ie the number of reconstructed instants is 256. It has to be set to a more reasonable
    // value which is why a "frames" argument is introduced
    auto inputSize = itk::MakeSize(constantImageSource->GetSize()[0],
                                   constantImageSource->GetSize()[1],
                                   constantImageSource->GetSize()[2],
                                   args_info.frames_arg);
    constantImageSource->SetSize(inputSize);

    inputFilter = constantImageSource;
  }
  TRY_AND_EXIT_ON_ITK_EXCEPTION(inputFilter->Update())
  inputFilter->ReleaseDataFlagOn();

  // Re-order geometry and projections
  // In the new order, projections with identical phases are packed together
  std::vector<double> signal = rtk::ReadSignalFile(args_info.signal_arg);
  using ReorderProjectionsFilterType = rtk::ReorderProjectionsImageFilter<ProjectionStackType>;
  auto reorder = ReorderProjectionsFilterType::New();
  reorder->SetInput(reader->GetOutput());
  reorder->SetInputGeometry(geometry);
  reorder->SetInputSignal(signal);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reorder->Update())

  // Release the memory holding the stack of original projections
  reader->GetOutput()->ReleaseData();

  // Compute the interpolation weights
  auto signalToInterpolationWeights = rtk::SignalToInterpolationWeights::New();
  signalToInterpolationWeights->SetSignal(reorder->GetOutputSignal());
  signalToInterpolationWeights->SetNumberOfReconstructedFrames(
    inputFilter->GetOutput()->GetLargestPossibleRegion().GetSize(3));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(signalToInterpolationWeights->Update())

  // Create the 4DROOSTER filter, connect the basic inputs, and set the basic parameters
  // Also set the forward and back projection filters to be used
  using ROOSTERFilterType = rtk::FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>;
  auto rooster = ROOSTERFilterType::New();
  SetForwardProjectionFromGgo(args_info, rooster.GetPointer());
  SetBackProjectionFromGgo(args_info, rooster.GetPointer());
  rooster->SetInputVolumeSeries(inputFilter->GetOutput());
  rooster->SetCG_iterations(args_info.cgiter_arg);
  rooster->SetMainLoop_iterations(args_info.niter_arg);
  rooster->SetPhaseShift(args_info.shift_arg);
  rooster->SetCudaConjugateGradient(args_info.cudacg_flag);
  rooster->SetUseCudaCyclicDeformation(args_info.cudadvfinterpolation_flag);
  rooster->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  REPORT_ITERATIONS(rooster, ROOSTERFilterType, VolumeSeriesType)

  // Set the newly ordered arguments
  rooster->SetInputProjectionStack(reorder->GetOutput());
  rooster->SetGeometry(reorder->GetOutputGeometry());
  rooster->SetWeights(signalToInterpolationWeights->GetOutput());
  rooster->SetSignal(reorder->GetOutputSignal());

  // For each optional regularization step, set whether or not
  // it should be performed, and provide the necessary inputs

  // Positivity
  if (args_info.nopositivity_flag)
    rooster->SetPerformPositivity(false);
  else
    rooster->SetPerformPositivity(true);

  // Motion mask
  VolumeType::Pointer motionMask;
  if (args_info.motionmask_given)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(motionMask = itk::ReadImage<VolumeType>(args_info.motionmask_arg))
    rooster->SetMotionMask(motionMask);
    rooster->SetPerformMotionMask(true);
  }
  else
    rooster->SetPerformMotionMask(false);

  // Spatial TV
  if (args_info.gamma_space_given)
  {
    rooster->SetGammaTVSpace(args_info.gamma_space_arg);
    rooster->SetTV_iterations(args_info.tviter_arg);
    rooster->SetPerformTVSpatialDenoising(true);
  }
  else
    rooster->SetPerformTVSpatialDenoising(false);

  // Spatial wavelets
  if (args_info.threshold_given)
  {
    rooster->SetSoftThresholdWavelets(args_info.threshold_arg);
    rooster->SetOrder(args_info.order_arg);
    rooster->SetNumberOfLevels(args_info.levels_arg);
    rooster->SetPerformWaveletsSpatialDenoising(true);
  }
  else
    rooster->SetPerformWaveletsSpatialDenoising(false);

  // Temporal TV
  if (args_info.gamma_time_given)
  {
    rooster->SetGammaTVTime(args_info.gamma_time_arg);
    rooster->SetTV_iterations(args_info.tviter_arg);
    rooster->SetPerformTVTemporalDenoising(true);
  }
  else
    rooster->SetPerformTVTemporalDenoising(false);

  // Temporal L0
  if (args_info.lambda_time_given)
  {
    rooster->SetLambdaL0Time(args_info.lambda_time_arg);
    rooster->SetL0_iterations(args_info.l0iter_arg);
    rooster->SetPerformL0TemporalDenoising(true);
  }
  else
    rooster->SetPerformL0TemporalDenoising(false);

  // Total nuclear variation
  if (args_info.gamma_tnv_given)
  {
    rooster->SetGammaTNV(args_info.gamma_tnv_arg);
    rooster->SetTV_iterations(args_info.tviter_arg);
    rooster->SetPerformTNVDenoising(true);
  }
  else
    rooster->SetPerformTNVDenoising(false);

  // Warping
  if (args_info.dvf_given)
  {
    rooster->SetPerformWarping(true);

    if (args_info.nn_flag)
      rooster->SetUseNearestNeighborInterpolationInWarping(true);

    // Read DVF
    DVFSequenceImageType::Pointer dvf;
    TRY_AND_EXIT_ON_ITK_EXCEPTION(dvf = itk::ReadImage<DVFSequenceImageType>(args_info.dvf_arg))
    rooster->SetDisplacementField(dvf);

    if (args_info.idvf_given)
    {
      rooster->SetComputeInverseWarpingByConjugateGradient(false);

      // Read inverse DVF if provided
      DVFSequenceImageType::Pointer idvf;
      TRY_AND_EXIT_ON_ITK_EXCEPTION(idvf = itk::ReadImage<DVFSequenceImageType>(args_info.idvf_arg))
      rooster->SetInverseDisplacementField(idvf);
    }
  }

  TRY_AND_EXIT_ON_ITK_EXCEPTION(rooster->Update())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(rooster->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
