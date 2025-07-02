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

#include "rtkdrawshepploganphantom_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkAdditiveGaussianNoiseImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>


int
main(int argc, char * argv[])
{
  GGO(rtkdrawshepploganphantom, args_info);

  constexpr unsigned int Dimension = 3;

  using OutputImageType = itk::Image<float, Dimension>;

  // Create a stack of empty projection images
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkdrawshepploganphantom>(constantImageSource,
                                                                                                  args_info);

  // Create a reference object (in this case a 3D phantom reference).
  using DSLType = rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType>;
  DSLType::VectorType offset(0.);
  DSLType::VectorType scale;
  for (unsigned int i = 0; i < std::min(args_info.offset_given, Dimension); i++)
    offset[i] = args_info.offset_arg[i];
  scale.Fill(args_info.phantomscale_arg[0]);
  for (unsigned int i = 0; i < std::min(args_info.phantomscale_given, Dimension); i++)
    scale[i] = args_info.phantomscale_arg[i];
  auto dsl = DSLType::New();
  dsl->SetPhantomScale(scale);
  dsl->SetInput(constantImageSource->GetOutput());
  dsl->SetOriginOffset(offset);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update())

  // Add noise
  OutputImageType::Pointer output = dsl->GetOutput();
  if (args_info.noise_given)
  {
    auto noisy = rtk::AdditiveGaussianNoiseImageFilter<OutputImageType>::New();
    noisy->SetInput(output);
    noisy->SetMean(0.0);
    noisy->SetStandardDeviation(args_info.noise_arg);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(noisy->Update())
    output = noisy->GetOutput();
  }

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(output, args_info.output_arg))

  return EXIT_SUCCESS;
}
