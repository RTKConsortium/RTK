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

#include "rtksirt_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkSIRTConeBeamReconstructionFilter.h"
#include "rtkNormalizedJosephBackProjectionImageFilter.h"
#if CUDA_FOUND
  #include "rtkCudaSIRTConeBeamReconstructionFilter.h"
  #include "itkCudaImage.h"
#endif

#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtksirt, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

#if CUDA_FOUND
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
#endif

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtksirt>(reader, args_info);

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
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtksirt>(constantImageSource, args_info);
    inputFilter = constantImageSource;
    }

  // Set the forward and back projection filters to be used
  typedef rtk::SIRTConeBeamReconstructionFilter<OutputImageType> SIRTFilterType;
  SIRTFilterType::Pointer sirt = SIRTFilterType::New();
  sirt->ConfigureForwardProjection(args_info.method_arg);
  sirt->ConfigureBackProjection(args_info.bp_arg);

  sirt->SetInput( inputFilter->GetOutput() );
  sirt->SetInput(1, reader->GetOutput());
  sirt->SetGeometry( geometryReader->GetOutputObject() );
  sirt->SetNumberOfIterations( args_info.niterations_arg );

  itk::TimeProbe readerProbe;
  if(args_info.time_flag)
    {
    std::cout << "Recording elapsed time... " << std::flush;
    readerProbe.Start();
    }

  TRY_AND_EXIT_ON_ITK_EXCEPTION( sirt->Update() )

  if(args_info.time_flag)
    {
//    sirt->PrintTiming(std::cout);
    readerProbe.Stop();
    std::cout << "It took...  " << readerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;
    }

  // Write
  typedef itk::ImageFileWriter< OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( sirt->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
