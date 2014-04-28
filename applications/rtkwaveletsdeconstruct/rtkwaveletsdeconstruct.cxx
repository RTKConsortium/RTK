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

#include "rtkwaveletsdeconstruct_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkDeconstructImageFilter.h"
#include "rtkReconstructImageFilter.h"
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <string>
#include <sstream>

int main(int argc, char * argv[])
{
  GGO(rtkwaveletsdeconstruct, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 2;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Read the input image
  typedef itk::ImageFileReader<OutputImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(args_info.input_arg);

  // Create the deconstruction filter
  typedef rtk::DeconstructImageFilter<OutputImageType> DeconstructFilterType;
  DeconstructFilterType::Pointer wavelets = DeconstructFilterType::New();
  wavelets->SetOrder(args_info.order_arg);
  wavelets->SetNumberOfLevels(args_info.level_arg);
  wavelets->SetInput(reader->GetOutput());
  wavelets->Update();

//  // Write deconstruction
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
//  for (unsigned int outputIndex=0; outputIndex<wavelets->GetNumberOfIndexedOutputs(); outputIndex++)
//    {
//    std::ostringstream os ;
//    os << outputIndex << ".mha";

//    writer->SetFileName(os.str());
//    writer->SetInput( wavelets->GetOutput(outputIndex) );
//    TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );
//    }

  // Reconstruct the original image
  typedef rtk::ReconstructImageFilter<OutputImageType> ReconstructFilterType;
  ReconstructFilterType::Pointer waveletsR = ReconstructFilterType::New();
  waveletsR->SetOrder(args_info.order_arg);
  waveletsR->SetNumberOfLevels(args_info.level_arg);
  for (unsigned int outputIndex=0; outputIndex<wavelets->GetNumberOfIndexedOutputs(); outputIndex++)
    {
    wavelets->GetOutput(outputIndex)->Print(std::cout);
    waveletsR->SetInput(outputIndex, wavelets->GetOutput(outputIndex));
    }
  waveletsR->SetSizes(wavelets->GetSizes());
  waveletsR->Update();

  // Write reconstruction
  std::ostringstream os ;
  os << "reconstruction.mha";
  writer->SetFileName(os.str());
  writer->SetInput( waveletsR->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
