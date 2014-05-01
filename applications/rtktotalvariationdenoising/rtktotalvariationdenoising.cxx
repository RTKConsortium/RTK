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
#include "rtkTotalVariationDenoisingBPDQImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#ifdef RTK_USE_CUDA
#include <itkCudaImage.h>
#endif

int main(int argc, char * argv[])
{
  GGO(rtktotalvariationdenoising, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
  typedef itk::CudaImage< itk::CovariantVector 
      < OutputPixelType, Dimension >, Dimension >                GradientOutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension >     OutputImageType;
  typedef itk::Image< itk::CovariantVector 
      < OutputPixelType, Dimension >, Dimension >                GradientOutputImageType;
#endif
  
  // Read input
  typedef itk::ImageFileReader<OutputImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( args_info.input_arg );

  // Apply total variation denoising
  typedef rtk::TotalVariationDenoisingBPDQImageFilter
      <OutputImageType, GradientOutputImageType> TVDenoisingFilterType;
  TVDenoisingFilterType::Pointer tv = TVDenoisingFilterType::New();
  tv->SetInput(reader->GetOutput());
  tv->SetGamma(args_info.gamma_arg);
  tv->SetNumberOfIterations(args_info.niter_arg);

  bool* dimsProcessed = new bool[Dimension];
  for (unsigned int i=0; i<Dimension; i++)
    {
    if ((args_info.dim_given) && (args_info.dim_arg[i] == 0)) dimsProcessed[i] = false;
    else dimsProcessed[i] = true;
//    std::cout << dimsProcessed[i];
    }
//  std::cout << std::endl;
  tv->SetDimensionsProcessed(dimsProcessed);

  // Write
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput(tv->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  delete[] dimsProcessed;
  return EXIT_SUCCESS;
}
