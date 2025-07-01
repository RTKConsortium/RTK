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

#include "rtkfourdconjugategradient_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkSignalToInterpolationWeights.h"
#include "rtkReorderProjectionsImageFilter.h"

#ifdef RTK_USE_CUDA
#  include "itkCudaImage.h"
#  include "rtkCudaConstantVolumeSeriesSource.h"
#endif

#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkfourdconjugategradient, args_info);

  using OutputPixelType = float;

#ifdef RTK_USE_CUDA
  using VolumeSeriesType = itk::CudaImage<OutputPixelType, 4>;
  using ProjectionStackType = itk::CudaImage<OutputPixelType, 3>;
#else
  using VolumeSeriesType = itk::Image<OutputPixelType, 4>;
  using ProjectionStackType = itk::Image<OutputPixelType, 3>;
#endif

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<ProjectionStackType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkfourdconjugategradient>(reader, args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->UpdateLargestPossibleRegion())

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
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkfourdconjugategradient>(
      constantImageSource, args_info);

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

  // Set the forward and back projection filters to be used
  using ConjugateGradientFilterType =
    rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>;
  auto conjugategradient = ConjugateGradientFilterType::New();
  SetForwardProjectionFromGgo(args_info, conjugategradient.GetPointer());
  SetBackProjectionFromGgo(args_info, conjugategradient.GetPointer());
  conjugategradient->SetInputVolumeSeries(inputFilter->GetOutput());
  conjugategradient->SetNumberOfIterations(args_info.niterations_arg);
  conjugategradient->SetCudaConjugateGradient(args_info.cudacg_flag);
  conjugategradient->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  // Set the newly ordered arguments
  conjugategradient->SetInputProjectionStack(reorder->GetOutput());
  conjugategradient->SetGeometry(reorder->GetOutputGeometry());
  conjugategradient->SetWeights(signalToInterpolationWeights->GetOutput());
  conjugategradient->SetSignal(reorder->GetOutputSignal());

  REPORT_ITERATIONS(conjugategradient, ConjugateGradientFilterType, VolumeSeriesType);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(conjugategradient->Update())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(conjugategradient->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
