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

#include "rtkspectraldenoiseprojections_ggo.h"
#include "rtkMacro.h"
#include "rtkGgoFunctions.h"
#include "rtkConditionalMedianImageFilter.h"

#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>

int
main(int argc, char * argv[])
{
  GGO(rtkspectraldenoiseprojections, args_info);

  constexpr unsigned int Dimension = 3;
  using OutputImageType = itk::VectorImage<float, Dimension>;

  // Reader
  OutputImageType::Pointer input;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(input = itk::ReadImage<OutputImageType>(args_info.input_arg))

  // Remove aberrant pixels
  using MedianType = rtk::ConditionalMedianImageFilter<OutputImageType>;
  auto median = MedianType::New();
  median->SetThresholdMultiplier(args_info.multiplier_arg);
  MedianType::MedianRadiusType radius;
  if (args_info.radius_given)
  {
    radius.Fill(args_info.radius_arg[0]);
    for (unsigned int i = 0; i < args_info.radius_given; i++)
      radius[i] = args_info.radius_arg[i];
  }
  median->SetRadius(radius);
  median->SetInput(input);

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(median->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
