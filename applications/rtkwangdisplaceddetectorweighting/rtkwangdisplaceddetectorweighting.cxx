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

#include "rtkwangdisplaceddetectorweighting_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkDisplacedDetectorImageFilter.h"
#ifdef RTK_USE_CUDA
#  include "rtkCudaDisplacedDetectorImageFilter.h"
#endif

#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkwangdisplaceddetectorweighting, args_info);

  // Check on hardware parameter
#ifndef RTK_USE_CUDA
  if (!strcmp(args_info.hardware_arg, "cuda"))
  {
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
  }
#endif

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkwangdisplaceddetectorweighting>(reader, args_info);

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

  // Displaced detector weighting
  using DDFCPUType = rtk::DisplacedDetectorImageFilter<OutputImageType>;
#ifdef RTK_USE_CUDA
  using DDFType = rtk::CudaDisplacedDetectorImageFilter;
#else
  using DDFType = rtk::DisplacedDetectorImageFilter<OutputImageType>;
#endif
  DDFCPUType::Pointer ddf;
  if (!strcmp(args_info.hardware_arg, "cuda"))
    ddf = DDFType::New();
  else
    ddf = DDFCPUType::New();
  ddf->SetInput(reader->GetOutput());
  ddf->SetGeometry(geometry);
  if (args_info.minOffset_given && args_info.maxOffset_given)
    ddf->SetOffsets(args_info.minOffset_arg, args_info.maxOffset_arg);

  // Write
  auto writer = itk::ImageFileWriter<OutputImageType>::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(ddf->GetOutput());
  writer->SetNumberOfStreamDivisions(args_info.divisions_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  return EXIT_SUCCESS;
}
