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

#include "rtktotalnuclearvariationdenoising_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "rtkTotalNuclearVariationDenoisingBPDQImageFilter.h"


#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtktotalnuclearvariationdenoising, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 4;           // Number of dimensions of the input image
  constexpr unsigned int DimensionsProcessed = 3; // Number of dimensions along which the gradient is computed

  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  using GradientOutputImageType = itk::Image<itk::CovariantVector<OutputPixelType, DimensionsProcessed>, Dimension>;
  using TVDenoisingFilterType =
    rtk::TotalNuclearVariationDenoisingBPDQImageFilter<OutputImageType, GradientOutputImageType>;

  // Read input
  OutputImageType::Pointer input;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(input = itk::ReadImage<OutputImageType>(args_info.input_arg))

  // Apply total nuclear variation denoising
  auto tv = TVDenoisingFilterType::New();
  tv->SetInput(input);
  tv->SetGamma(args_info.gamma_arg);
  tv->SetNumberOfIterations(args_info.niter_arg);

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(tv->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
