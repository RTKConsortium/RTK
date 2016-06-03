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

#include "rtktutorialapplication_ggo.h"
#include "rtkGgoFunctions.h"

#include <itkAddImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtktutorialapplication, args_info);

  // This small application can be used by RTK beginners
  // as a starting point. It reads a volume and a scalar,
  // adds the scalar to all voxels of the volume, and
  // writes the result

  // Below is a list of a few possible challenges for beginners:

  // Copy/paste the folder of this application to a new one, named
  // "rtkMyApp", and modify the following files
  //      -> applications/CMakeLists.txt
  //      -> applications/rtkMyApp/rtkMyApp.cxx
  //      -> applications/rtkMyApp/rtkMyApp.ggo
  //      -> applications/rtkMyApp/CMakeLists.txt
  // so that rtkMyApp compiles and runs fine.

  // Modify the following files
  //      -> applications/rtkMyApp/rtkMyApp.cxx
  //      -> applications/rtkMyApp/rtkMyApp.ggo
  // so that rtkMyApp takes two volumes in input, adds them,
  // and writes the result in output

  // Modify the following files
  //      -> applications/rtkMyApp/rtkMyApp.cxx
  //      -> applications/rtkMyApp/rtkMyApp.ggo
  // so that rtkMyApp takes a volume in input, adds a scalar to
  // all voxels, multiplies the output by an other scalar, and
  // writes the result in output

  // !! HARDER !!
  // Modify the following files
  //      -> applications/rtkMyApp/rtkMyApp.cxx
  // so that rtkMyApp computes the n-th element of the
  // Collatz sequence of each voxel, using only one
  // itk::AddImageFilter, one itk::MultiplyImageFilter,
  // and one itk::DivideImageFilter
  // http://en.wikipedia.org/wiki/Collatz_conjecture
  //
  // You will need to use the DisconnectPipeline()
  // function. You can see how it is used in
  // rtkSARTConeBeamReconstructionFilter.hxx

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Read the input volume
  typedef itk::ImageFileReader<  OutputImageType > InputReaderType;
  InputReaderType::Pointer inputReader = InputReaderType::New();
  inputReader->SetFileName( args_info.input_arg );

  // Create the Add filter
  typedef itk::AddImageFilter<OutputImageType> AddFilterType;
  AddFilterType::Pointer add = AddFilterType::New();
  add->SetInput1(inputReader->GetOutput());
  add->SetConstant2( args_info.constant_arg );

  TRY_AND_EXIT_ON_ITK_EXCEPTION( add->Update() )

  // Write
  typedef itk::ImageFileWriter< OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( add->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
