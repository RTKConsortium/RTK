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

#include "rtkprojections_ggo.h"
#include "rtkMacro.h"
#include "rtkGgoFunctions.h"

#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkprojections, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkprojections>(reader, args_info);

  // Write
  using WriterType = itk::ImageFileWriter<OutputImageType>;
  auto writer = WriterType::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(reader->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->UpdateOutputInformation())
  writer->SetNumberOfStreamDivisions(1 + reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() /
                                           (1024 * 1024 * 4));

  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  return EXIT_SUCCESS;
}
