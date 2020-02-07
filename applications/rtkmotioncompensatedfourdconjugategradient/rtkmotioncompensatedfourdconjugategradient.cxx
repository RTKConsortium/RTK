/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtkmotioncompensatedfourdconjugategradient_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkPhasesToInterpolationWeights.h"
#include "rtkWarpSequenceImageFilter.h"

#ifdef RTK_USE_CUDA
#  include "itkCudaImage.h"
#endif
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkmotioncompensatedfourdconjugategradient, args_info);

  using OutputPixelType = float;
  using DVFVectorType = itk::CovariantVector<OutputPixelType, 3>;

#ifdef RTK_USE_CUDA
  using VolumeSeriesType = itk::CudaImage<OutputPixelType, 4>;
  using ProjectionStackType = itk::CudaImage<OutputPixelType, 3>;
  using DVFSequenceImageType = itk::CudaImage<DVFVectorType, VolumeSeriesType::ImageDimension>;
  using DVFImageType = itk::CudaImage<DVFVectorType, VolumeSeriesType::ImageDimension - 1>;
#else
  using VolumeSeriesType = itk::Image<OutputPixelType, 4>;
  using ProjectionStackType = itk::Image<OutputPixelType, 3>;
  using DVFSequenceImageType = itk::Image<DVFVectorType, VolumeSeriesType::ImageDimension>;
  using DVFImageType = itk::Image<DVFVectorType, VolumeSeriesType::ImageDimension - 1>;
#endif
  using DVFReaderType = itk::ImageFileReader<DVFSequenceImageType>;

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<ProjectionStackType>;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkmotioncompensatedfourdconjugategradient>(reader, args_info);

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometryReader->GenerateOutputInformation())

  // Create input: either an existing volume read from a file or a blank image
  itk::ImageSource<VolumeSeriesType>::Pointer inputFilter;
  if (args_info.input_given)
  {
    // Read an existing image to initialize the volume
    using InputReaderType = itk::ImageFileReader<VolumeSeriesType>;
    InputReaderType::Pointer inputReader = InputReaderType::New();
    inputReader->SetFileName(args_info.input_arg);
    inputFilter = inputReader;
  }
  else
  {
    // Create new empty volume
    using ConstantImageSourceType = rtk::ConstantImageSource<VolumeSeriesType>;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkmotioncompensatedfourdconjugategradient>(
      constantImageSource, args_info);

    // GenGetOpt can't handle default arguments for multiple arguments like dimension or spacing.
    // The only default it accepts is to set all components of a multiple argument to the same value.
    // Default dimension is 256^4, ie the number of reconstructed instants is 256. It has to be set to a more reasonable
    // value which is why a "frames" argument is introduced
    ConstantImageSourceType::SizeType inputSize = constantImageSource->GetSize();
    inputSize[3] = args_info.frames_arg;
    constantImageSource->SetSize(inputSize);

    inputFilter = constantImageSource;
  }
  TRY_AND_EXIT_ON_ITK_EXCEPTION(inputFilter->Update())
  inputFilter->ReleaseDataFlagOn();

  // Read the phases file
  rtk::PhasesToInterpolationWeights::Pointer phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(args_info.signal_arg);
  phaseReader->SetNumberOfReconstructedFrames(inputFilter->GetOutput()->GetLargestPossibleRegion().GetSize(3));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(phaseReader->Update())


  // Create the mcfourdcg filter, connect the basic inputs, and set the basic parameters
  using MCFourDCGFilterType =
    rtk::MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>;
  MCFourDCGFilterType::Pointer mcfourdcg = MCFourDCGFilterType::New();
  mcfourdcg->SetInputVolumeSeries(inputFilter->GetOutput());
  mcfourdcg->SetInputProjectionStack(reader->GetOutput());
  mcfourdcg->SetGeometry(geometryReader->GetOutputObject());
  mcfourdcg->SetWeights(phaseReader->GetOutput());
  mcfourdcg->SetNumberOfIterations(args_info.niter_arg);
  mcfourdcg->SetCudaConjugateGradient(args_info.cudacg_flag);
  mcfourdcg->SetSignal(rtk::ReadSignalFile(args_info.signal_arg));
  mcfourdcg->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  REPORT_ITERATIONS(mcfourdcg, MCFourDCGFilterType, VolumeSeriesType)

  // Read DVF
  DVFReaderType::Pointer dvfReader = DVFReaderType::New();
  dvfReader->SetFileName(args_info.dvf_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dvfReader->Update())
  mcfourdcg->SetDisplacementField(dvfReader->GetOutput());

  // Read inverse DVF if provided
  DVFReaderType::Pointer idvfReader = DVFReaderType::New();
  idvfReader->SetFileName(args_info.idvf_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(idvfReader->Update())
  mcfourdcg->SetInverseDisplacementField(idvfReader->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(mcfourdcg->Update())

  // The mcfourdcg filter reconstructs a static 4D volume (if the DVFs perfectly model the actual motion)
  // Warp this sequence with the inverse DVF so as to obtain a result similar to the classical 4D CG filter
  using WarpSequenceFilterType =
    rtk::WarpSequenceImageFilter<VolumeSeriesType, DVFSequenceImageType, ProjectionStackType, DVFImageType>;
  WarpSequenceFilterType::Pointer warp = WarpSequenceFilterType::New();
  warp->SetInput(mcfourdcg->GetOutput());
  warp->SetDisplacementField(idvfReader->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(warp->Update())

  // Write
  using WriterType = itk::ImageFileWriter<VolumeSeriesType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(warp->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  return EXIT_SUCCESS;
}
