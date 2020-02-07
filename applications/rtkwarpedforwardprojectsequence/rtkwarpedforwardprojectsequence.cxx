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

#include "rtkwarpedforwardprojectsequence_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkGeneralPurposeFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
//#include "rtkWarpForwardProjectSequenceImageFilter.h"
#include "rtkWarpFourDToProjectionStackImageFilter.h"
#include "rtkPhasesToInterpolationWeights.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkwarpedforwardprojectsequence, args_info);

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
  using DVFReaderType = itk::ImageFileReader<DVFSequenceImageType>;

  // Create a stack of empty projection images
  using ConstantImageSourceType = rtk::ConstantImageSource<ProjectionStackType>;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkwarpedforwardprojectsequence>(
    constantImageSource, args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantImageSource->Update())

  // Read the input volume sequence
  using volumeSeriesReaderType = itk::ImageFileReader<VolumeSeriesType>;
  volumeSeriesReaderType::Pointer volumeSeriesReader = volumeSeriesReaderType::New();
  volumeSeriesReader->SetFileName(args_info.input_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(volumeSeriesReader->Update())

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometryReader->GenerateOutputInformation())

  // Read the phases file
  rtk::PhasesToInterpolationWeights::Pointer phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(args_info.signal_arg);
  phaseReader->SetNumberOfReconstructedFrames(volumeSeriesReader->GetOutput()->GetLargestPossibleRegion().GetSize(3));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(phaseReader->Update())

  // Read DVF
  DVFReaderType::Pointer dvfReader = DVFReaderType::New();
  dvfReader->SetFileName(args_info.dvf_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dvfReader->Update())

  if (args_info.verbose_flag)
    std::cout << "Projecting volume sequence..." << std::endl;

  using WarpForwardProjectType = rtk::WarpFourDToProjectionStackImageFilter<VolumeSeriesType, ProjectionStackType>;
  WarpForwardProjectType::Pointer forwardProjection = WarpForwardProjectType::New();

  forwardProjection->SetInputProjectionStack(constantImageSource->GetOutput());
  forwardProjection->SetInputVolumeSeries(volumeSeriesReader->GetOutput());
  forwardProjection->SetDisplacementField(dvfReader->GetOutput());
  forwardProjection->SetGeometry(geometryReader->GetOutputObject());
  forwardProjection->SetWeights(phaseReader->GetOutput());
  forwardProjection->SetSignal(rtk::ReadSignalFile(args_info.signal_arg));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(forwardProjection->Update())

  // Write
  if (args_info.verbose_flag)
    std::cout << "Writing... " << std::endl;
  using WriterType = itk::ImageFileWriter<ProjectionStackType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(forwardProjection->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  return EXIT_SUCCESS;
}
