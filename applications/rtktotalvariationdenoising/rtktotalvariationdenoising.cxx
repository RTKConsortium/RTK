/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
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
  #include "rtkCudaTotalVariationDenoisingBPDQImageFilter.h"
#else
  #include "rtkTotalVariationDenoisingBPDQImageFilter.h"
#endif

#include "rtkTotalVariationImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtktotalvariationdenoising, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3; // Number of dimensions of the input image

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension >          OutputImageType;
  typedef itk::CudaImage< itk::CovariantVector
      < OutputPixelType, Dimension >, Dimension >               GradientOutputImageType;
  typedef rtk::CudaTotalVariationDenoisingBPDQImageFilter       TVDenoisingFilterType;
#else
  typedef itk::Image< OutputPixelType, Dimension >              OutputImageType;
  typedef itk::Image< itk::CovariantVector 
      < OutputPixelType, Dimension >, Dimension >               GradientOutputImageType;
  typedef rtk::TotalVariationDenoisingBPDQImageFilter
      <OutputImageType, GradientOutputImageType>                TVDenoisingFilterType;
#endif
  
  // Read input
  typedef itk::ImageFileReader<OutputImageType>                 ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( args_info.input_arg );
  reader->Update();

  // Compute total variation before denoising
  typedef rtk::TotalVariationImageFilter<OutputImageType>       TVFilterType;
  TVFilterType::Pointer tv = TVFilterType::New();
  tv->SetInput(reader->GetOutput());
  if (args_info.verbose_flag)
    {
    tv->Update();
    std::cout << "TV before denoising = " << tv->GetTotalVariation() << std::endl;
    }

  // Apply total variation denoising
  TVDenoisingFilterType::Pointer tvdenoising = TVDenoisingFilterType::New();
  tvdenoising->SetInput(reader->GetOutput());
  tvdenoising->SetGamma(args_info.gamma_arg);
  tvdenoising->SetNumberOfIterations(args_info.niter_arg);

  // Write
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput(tvdenoising->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  // Compute total variation after denoising
  if (args_info.verbose_flag)
    {
    tv->SetInput(tvdenoising->GetOutput());
    tv->Update();
    std::cout << "TV after denoising = " << tv->GetTotalVariation() << std::endl;
    }

  return EXIT_SUCCESS;
}
