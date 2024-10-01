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

#include "rtkkatsevichrecons_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "rtkThreeDHelicalProjectionGeometryXMLFileReader.h"
#include "rtkCyclicDeformationImageFilter.h"
#include "rtkProgressCommands.h"

#include <itkStreamingImageFilter.h>
#include <itkImageRegionSplitterDirection.h>
#include <itkImageFileWriter.h>

#include <rtkKatsevichConeBeamReconstructionFilter.h>


int
main(int argc, char * argv[])
{
  GGO(rtkkatsevichrecons, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

  using CPUOutputImageType = itk::Image<OutputPixelType, Dimension>;
#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = CPUOutputImageType;
#endif

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkkatsevichrecons>(reader, args_info);

  if (!args_info.lowmem_flag)
  {
    if (args_info.verbose_flag)
      std::cout << "Reading... " << std::endl;
    TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update())
  }

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDHelicalProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDHelicalProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometryReader->GenerateOutputInformation())
  rtk::ThreeDHelicalProjectionGeometry::Pointer geometry;
  geometry = geometryReader->GetOutputObject();
  geometry->VerifyHelixParameters();

  // Create reconstructed image
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkkatsevichrecons>(constantImageSource,
                                                                                            args_info);

  using KatsevichReconsFilterType = rtk::KatsevichConeBeamReconstructionFilter<OutputImageType>;
  KatsevichReconsFilterType::Pointer kats = KatsevichReconsFilterType::New();
  kats->SetGeometry(geometry);
  kats->SetInput(0, constantImageSource->GetOutput());
  kats->SetInput(1, reader->GetOutput());


// Check on hardware parameter
#ifndef RTK_USE_CUDA
  if (!strcmp(args_info.hardware_arg, "cuda"))
  {
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
  }
#endif

  //// Streaming depending on streaming capability of writer
  // using StreamerType = itk::StreamingImageFilter<CPUOutputImageType, CPUOutputImageType>;
  // StreamerType::Pointer streamerBP = StreamerType::New();
  // streamerBP->SetInput(bp->GetOutput());
  // streamerBP->SetNumberOfStreamDivisions(args_info.divisions_arg);
  // itk::ImageRegionSplitterDirection::Pointer splitter = itk::ImageRegionSplitterDirection::New();
  // splitter->SetDirection(2); // Prevent splitting along z axis. As a result, splitting will be performed along y axis
  // streamerBP->SetRegionSplitter(splitter);

  // Write
  using WriterType = itk::ImageFileWriter<CPUOutputImageType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(kats->GetOutput());

  if (args_info.verbose_flag)
    std::cout << "Reconstructing and writing... " << std::endl;

  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  return EXIT_SUCCESS;
}
