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

#include "rtkfdktwodweights_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkFDKWeightProjectionFilter.h"

#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkfdktwodweights, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkfdktwodweights>(reader, args_info);

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

  // Weights filter
  using WeightType = rtk::FDKWeightProjectionFilter<OutputImageType>;
  auto wf = WeightType::New();
  wf->SetInput(reader->GetOutput());
  wf->SetGeometry(geometry);
  wf->SetNumberOfWorkUnits(1);
  wf->InPlaceOff();

  // Write
  using WriterType = itk::ImageFileWriter<OutputImageType>;
  auto writer = WriterType::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(wf->GetOutput());
  writer->SetNumberOfStreamDivisions(args_info.divisions_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  return EXIT_SUCCESS;
}
