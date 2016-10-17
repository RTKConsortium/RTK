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

#include "rtktotalnuclearvariationdenoising_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "rtkTotalNuclearVariationDenoisingBPDQImageFilter.h"


#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtktotalnuclearvariationdenoising, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 4; // Number of dimensions of the input image
  const unsigned int DimensionsProcessed = 3; // Number of dimensions along which the gradient is computed

  typedef itk::Image< OutputPixelType, Dimension >     OutputImageType;
  typedef itk::Image< itk::CovariantVector 
      < OutputPixelType, DimensionsProcessed >, Dimension >                GradientOutputImageType;
  typedef rtk::TotalNuclearVariationDenoisingBPDQImageFilter
      <OutputImageType, GradientOutputImageType>                           TVDenoisingFilterType;
  
  // Read input
  typedef itk::ImageFileReader<OutputImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( args_info.input_arg );
//  reader->ReleaseDataFlagOn();

  // Apply total nuclear variation denoising
  TVDenoisingFilterType::Pointer tv = TVDenoisingFilterType::New();
  tv->SetInput(reader->GetOutput());
  tv->SetGamma(args_info.gamma_arg);
  tv->SetNumberOfIterations(args_info.niter_arg);
//  tv->ReleaseDataFlagOn();

  // Write
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput(tv->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
