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
#include "rtkGgoFunctions.h"
#include "rtkMacro.h"

#include <itkImageFileWriter.h>
#include <itkImageIOFactory.h>
#include <itkVectorImage.h>

#include <iostream>

int
main(int argc, char * argv[])
{
  GGO(rtkprojections, args_info);

  constexpr unsigned int Dimension = 3;
  bool                   inputIsVectorImage = false;
  TRY_AND_EXIT_ON_ITK_EXCEPTION({
    const auto fileNames = rtk::GetProjectionsFileNamesFromGgo(args_info);
    if (!args_info.component_given && !fileNames.empty())
    {
      auto imageIO =
        itk::ImageIOFactory::CreateImageIO(fileNames.front().c_str(), itk::ImageIOFactory::IOFileModeEnum::ReadMode);
      if (imageIO.IsNotNull())
      {
        imageIO->SetFileName(fileNames.front());
        imageIO->ReadImageInformation();
        inputIsVectorImage = imageIO->GetPixelType() == itk::IOPixelEnum::VECTOR;
      }
    }
  })
  if (inputIsVectorImage)
  {
    using OutputImageType = itk::VectorImage<float, Dimension>;
    if (args_info.poisson_given || args_info.gaussian_given)
    {
      std::cerr << "Noise addition is not supported for vector projections" << std::endl;
      return EXIT_FAILURE;
    }

    using ReaderType = rtk::ProjectionsReader<OutputImageType>;
    auto reader = ReaderType::New();
    rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkprojections>(reader, args_info);

    auto writer = itk::ImageFileWriter<OutputImageType>::New();
    writer->SetFileName(args_info.output_arg);
    writer->SetInput(reader->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->UpdateOutputInformation())
    writer->SetNumberOfStreamDivisions(1 + reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() /
                                             (1024 * 1024 * 4));

    TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())
  }
  else
  {
    using OutputImageType = itk::Image<float, Dimension>;

    using ReaderType = rtk::ProjectionsReader<OutputImageType>;
    auto reader = ReaderType::New();
    rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkprojections>(reader, args_info);

    OutputImageType::Pointer output =
      rtk::AddNoiseFromGgo<OutputImageType, args_info_rtkprojections>(reader->GetOutput(), args_info);

    auto writer = itk::ImageFileWriter<OutputImageType>::New();
    writer->SetFileName(args_info.output_arg);
    writer->SetInput(output);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->UpdateOutputInformation())
    writer->SetNumberOfStreamDivisions(1 + reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() /
                                             (1024 * 1024 * 4));

    TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())
  }

  return EXIT_SUCCESS;
}
