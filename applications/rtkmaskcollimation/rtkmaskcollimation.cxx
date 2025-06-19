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

#include "rtkmaskcollimation_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkMaskCollimationImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkmaskcollimation, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkmaskcollimation>(reader, args_info);

  // Create projection image filter
  using OFMType = rtk::MaskCollimationImageFilter<OutputImageType, OutputImageType>;
  auto ofm = OFMType::New();
  ofm->SetInput(reader->GetOutput());
  ofm->SetGeometry(geometry);

  // Write
  using WriterType = itk::ImageFileWriter<OutputImageType>;
  auto writer = WriterType::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(ofm->GetOutput());
  if (args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->UpdateOutputInformation())
  writer->SetNumberOfStreamDivisions(reader->GetOutput()->GetLargestPossibleRegion().GetSize(2));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())
  return EXIT_SUCCESS;
}
