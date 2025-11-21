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
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMultiplyImageFilter.h>
#include <itkExpImageFilter.h>
#include <itkShotNoiseImageFilter.h>
#include <itkThresholdImageFilter.h>
#include <itkLogImageFilter.h>

#include "rtkaddpoissonnoise_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

int
main(int argc, char * argv[])
{
  GGO(rtkaddpoissonnoise, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

#ifdef RTK_USE_CUDA
  using ImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using ImageType = itk::Image<OutputPixelType, Dimension>;
#endif

  auto reader = itk::ImageFileReader<ImageType>::New();
  reader->SetFileName(args_info.input_arg);

  // Use ITK to add pre-log Poisson noise
  auto multiply = itk::MultiplyImageFilter<ImageType>::New();
  multiply->SetInput(reader->GetOutput());
  multiply->SetConstant(-args_info.muref_arg);

  auto exp = itk::ExpImageFilter<ImageType, ImageType>::New();
  exp->SetInput(multiply->GetOutput());

  auto multiply2 = itk::MultiplyImageFilter<ImageType>::New();
  multiply2->SetInput(exp->GetOutput());
  multiply2->SetConstant(args_info.I0_arg);

  auto poisson = itk::ShotNoiseImageFilter<ImageType>::New();
  poisson->SetInput(multiply2->GetOutput());

  auto threshold = itk::ThresholdImageFilter<ImageType>::New();
  threshold->SetInput(poisson->GetOutput());
  threshold->SetLower(1.);
  threshold->SetOutsideValue(1.);

  auto multiply3 = itk::MultiplyImageFilter<ImageType>::New();
  multiply3->SetInput(threshold->GetOutput());
  multiply3->SetConstant(1. / args_info.I0_arg);

  auto log = itk::LogImageFilter<ImageType, ImageType>::New();
  log->SetInput(multiply3->GetOutput());

  auto multiply4 = itk::MultiplyImageFilter<ImageType>::New();
  multiply4->SetInput(log->GetOutput());
  multiply4->SetConstant(-1. / args_info.muref_arg);

  itk::WriteImage(multiply4->GetOutput(), args_info.output_arg);

  return EXIT_SUCCESS;
}
