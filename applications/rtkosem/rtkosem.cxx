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

#include "rtkosem_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkOSEMConeBeamReconstructionFilter.h"
#include "rtkNormalizedJosephBackProjectionImageFilter.h"
#include "rtkPhaseGatingImageFilter.h"

#ifdef RTK_USE_CUDA
  #include "itkCudaImage.h"
#endif

#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkosem, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
#endif

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkosem>(reader, args_info);

  // Geometry
  if(args_info.verbose_flag)
    std::cout << "Reading geometry information from "
              << args_info.geometry_arg
              << "..."
              << std::endl;
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() )

  // Create input: either an existing volume read from a file or a blank image
  itk::ImageSource< OutputImageType >::Pointer inputFilter;
  if(args_info.input_given)
    {
    // Read an existing image to initialize the volume
    typedef itk::ImageFileReader<  OutputImageType > InputReaderType;
    InputReaderType::Pointer inputReader = InputReaderType::New();
    inputReader->SetFileName( args_info.input_arg );
    inputFilter = inputReader;
    }
  else
    {
    // Create new empty volume
    typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkosem>(constantImageSource, args_info);
    constantImageSource->SetConstant(1.);
    inputFilter = constantImageSource;
    }

  // OSEM reconstruction filter
  rtk::OSEMConeBeamReconstructionFilter< OutputImageType >::Pointer osem =
      rtk::OSEMConeBeamReconstructionFilter< OutputImageType >::New();

  // Set the forward and back projection filters
  osem->SetForwardProjectionFilter(args_info.fp_arg);
  osem->SetBackProjectionFilter(args_info.bp_arg);
  osem->SetInput( inputFilter->GetOutput() );
  osem->SetInput(1, reader->GetOutput());
  osem->SetGeometry( geometryReader->GetOutputObject() );

  osem->SetNumberOfIterations( args_info.niterations_arg );
  osem->SetNumberOfProjectionsPerSubset( args_info.nprojpersubset_arg );

  // Write
  typedef itk::ImageFileWriter< OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( osem->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
