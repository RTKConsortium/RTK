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

#include "rtkkatsevichderivative_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "rtkThreeDHelicalProjectionGeometryXMLFileReader.h"
#include "rtkKatsevichDerivativeImageFilter.h"
#include "rtkProgressCommands.h"

#include <itkStreamingImageFilter.h>
#include <itkImageRegionSplitterDirection.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkkatsevichderivative, args_info);

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
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkkatsevichderivative>(reader, args_info);

  if (args_info.verbose_flag)
    std::cout << "Reading... " << std::endl;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update())

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

  // Check on hardware parameter
#ifndef RTK_USE_CUDA
  if (!strcmp(args_info.hardware_arg, "cuda"))
  {
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
  }
#endif

  using KatsevichDerivativeType = rtk::KatsevichDerivativeImageFilter<OutputImageType, OutputImageType>;
  KatsevichDerivativeType::Pointer deriv = KatsevichDerivativeType::New();
  deriv->SetGeometry(geometry);
  deriv->SetInput(reader->GetOutput());

  // Write
  using WriterType = itk::ImageFileWriter<CPUOutputImageType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(deriv->GetOutput());

  if (args_info.verbose_flag)
    std::cout << "Reconstructing and writing... " << std::endl;

  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  return EXIT_SUCCESS;
}
