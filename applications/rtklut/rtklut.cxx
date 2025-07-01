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

#include "rtklut_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkLookupTableImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtklut, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtklut>(reader, args_info);

  // Read lookup table
  using LUTType = itk::Image<OutputPixelType, 1>;
  using LUTReaderType = itk::ImageFileReader<LUTType>;
  auto lutReader = LUTReaderType::New();
  lutReader->SetFileName(args_info.lut_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(lutReader->Update())

  // Apply lookup table
  using LUTFilterType = rtk::LookupTableImageFilter<OutputImageType, OutputImageType>;
  auto lutFilter = LUTFilterType::New();
  lutFilter->SetInput(reader->GetOutput());
  lutFilter->SetLookupTable(lutReader->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(lutFilter->Update())

  // Write
  if (args_info.verbose_flag)
    std::cout << "Writing result... " << std::endl;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(lutFilter->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
