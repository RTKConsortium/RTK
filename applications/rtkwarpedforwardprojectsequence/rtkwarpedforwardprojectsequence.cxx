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

#include "rtkwarpedforwardprojectsequence_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkGeneralPurposeFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
// #include "rtkWarpForwardProjectSequenceImageFilter.h"
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

  // Create a stack of empty projection images
  using ConstantImageSourceType = rtk::ConstantImageSource<ProjectionStackType>;
  auto constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkwarpedforwardprojectsequence>(
    constantImageSource, args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantImageSource->Update())

  // Read the input volume sequence
  VolumeSeriesType::Pointer volumeSeries;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(volumeSeries = itk::ReadImage<VolumeSeriesType>(args_info.input_arg))

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

  // Read the phases file
  auto phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(args_info.signal_arg);
  phaseReader->SetNumberOfReconstructedFrames(volumeSeries->GetLargestPossibleRegion().GetSize(3));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(phaseReader->Update())

  // Read DVF
  DVFSequenceImageType::Pointer dvf;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dvf = itk::ReadImage<DVFSequenceImageType>(args_info.dvf_arg))

  if (args_info.verbose_flag)
    std::cout << "Projecting volume sequence..." << std::endl;

  using WarpForwardProjectType = rtk::WarpFourDToProjectionStackImageFilter<VolumeSeriesType, ProjectionStackType>;
  auto forwardProjection = WarpForwardProjectType::New();

  forwardProjection->SetInputProjectionStack(constantImageSource->GetOutput());
  forwardProjection->SetInputVolumeSeries(volumeSeries);
  forwardProjection->SetDisplacementField(dvf);
  forwardProjection->SetGeometry(geometry);
  forwardProjection->SetWeights(phaseReader->GetOutput());
  forwardProjection->SetSignal(rtk::ReadSignalFile(args_info.signal_arg));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(forwardProjection->Update())

  // Write
  if (args_info.verbose_flag)
    std::cout << "Writing... " << std::endl;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(forwardProjection->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
