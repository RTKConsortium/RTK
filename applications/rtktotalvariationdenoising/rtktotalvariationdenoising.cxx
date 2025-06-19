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

#include "rtktotalvariationdenoising_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#ifdef RTK_USE_CUDA
#  include "rtkCudaTotalVariationDenoisingBPDQImageFilter.h"
#else
#  include "rtkTotalVariationDenoisingBPDQImageFilter.h"
#endif

#include "rtkTotalVariationImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtktotalvariationdenoising, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3; // Number of dimensions of the input image

#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
  using TVDenoisingFilterType = rtk::CudaTotalVariationDenoisingBPDQImageFilter;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  using GradientOutputImageType = itk::Image<itk::CovariantVector<OutputPixelType, Dimension>, Dimension>;
  using TVDenoisingFilterType = rtk::TotalVariationDenoisingBPDQImageFilter<OutputImageType, GradientOutputImageType>;
#endif

  // Read input
  OutputImageType::Pointer input;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(input = itk::ReadImage<OutputImageType>(args_info.input_arg))

  // Compute total variation before denoising
  using TVFilterType = rtk::TotalVariationImageFilter<OutputImageType>;
  auto tv = TVFilterType::New();
  tv->SetInput(input);
  if (args_info.verbose_flag)
  {
    tv->Update();
    std::cout << "TV before denoising = " << tv->GetTotalVariation() << std::endl;
  }

  // Apply total variation denoising
  auto tvdenoising = TVDenoisingFilterType::New();
  tvdenoising->SetInput(input);
  tvdenoising->SetGamma(args_info.gamma_arg);
  tvdenoising->SetNumberOfIterations(args_info.niter_arg);

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(tvdenoising->GetOutput(), args_info.output_arg))

  // Compute total variation after denoising
  if (args_info.verbose_flag)
  {
    tv->SetInput(tvdenoising->GetOutput());
    tv->Update();
    std::cout << "TV after denoising = " << tv->GetTotalVariation() << std::endl;
  }

  return EXIT_SUCCESS;
}
