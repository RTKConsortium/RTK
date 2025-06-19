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

#include "rtklastdimensionl0gradientdenoising_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

// #ifdef RTK_USE_CUDA
//   #include "rtkCudaLastDimensionL0GradientDenoisingImageFilter.h"
// #else
#include "rtkLastDimensionL0GradientDenoisingImageFilter.h"
// #endif

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtklastdimensionl0gradientdenoising, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 4; // Number of dimensions of the input image

  // #ifdef RTK_USE_CUDA
  //   using OutputImageType = itk::CudaImage< OutputPixelType, Dimension >;
  //   using DenoisingFilterType = rtk::CudaLastDimensionL0GradientDenoisingImageFilter;
  // #else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  using DenoisingFilterType = rtk::LastDimensionL0GradientDenoisingImageFilter<OutputImageType>;
  // #endif

  // Read input
  using ReaderType = itk::ImageFileReader<OutputImageType>;
  auto reader = ReaderType::New();
  reader->SetFileName(args_info.input_arg);
  reader->ReleaseDataFlagOn();

  // Apply L0 gradient norm denoising
  auto denoising = DenoisingFilterType::New();
  denoising->SetInput(reader->GetOutput());
  denoising->SetLambda(args_info.lambda_arg);
  denoising->SetNumberOfIterations(args_info.niter_arg);
  denoising->ReleaseDataFlagOn();

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(denoising->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
