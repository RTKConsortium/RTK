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

#include "rtkwarpedforwardprojections_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#ifdef RTK_USE_CUDA
#include "rtkCudaWarpedForwardProjectionImageFilter.h"
#endif

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkTimeProbe.h>

int main(int argc, char * argv[])
{
  GGO(rtkwarpedforwardprojections, args_info);

  const unsigned int Dimension = 3;
  typedef float OutputPixelType;
  typedef itk::CovariantVector< OutputPixelType, 3 > DVFVectorType;

  #ifdef RTK_USE_CUDA
    typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
    typedef itk::CudaImage<DVFVectorType, Dimension> DVFImageType;
  #else
    typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
    typedef itk::Image<DVFVectorType, Dimension> DVFImageType;
  #endif
  typedef itk::ImageFileReader< DVFImageType > DVFReaderType;

  // Geometry
  if(args_info.verbose_flag)
    std::cout << "Reading geometry information from "
              << args_info.geometry_arg
              << "..."
              << std::flush;
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() )
  if(args_info.verbose_flag)
    std::cout << " done." << std::endl;
  
  // Create a stack of empty projection images
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkwarpedforwardprojections>(constantImageSource, args_info);

  // Adjust size according to geometry
  ConstantImageSourceType::SizeType sizeOutput;
  sizeOutput[0] = constantImageSource->GetSize()[0];
  sizeOutput[1] = constantImageSource->GetSize()[1];
  sizeOutput[2] = geometryReader->GetOutputObject()->GetGantryAngles().size();
  constantImageSource->SetSize( sizeOutput );
  constantImageSource->Update();

  // Input reader
  if(args_info.verbose_flag)
    std::cout << "Reading input volume "
              << args_info.input_arg
              << "..."
              << std::flush;
  itk::TimeProbe readerProbe;
  typedef itk::ImageFileReader<  OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( args_info.input_arg );
  readerProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )
  readerProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << readerProbe.GetMean() << ' ' << readerProbe.GetUnit()
              << '.' << std::endl;
              
  // Create forward projection image filter
  if(args_info.verbose_flag)
    std::cout << "Projecting volume..." << std::flush;
  itk::TimeProbe projProbe;
  
  // Read DVF
  DVFReaderType::Pointer dvfReader = DVFReaderType::New();
  dvfReader->SetFileName( args_info.dvf_arg );
  dvfReader->Update();
  
#ifdef RTK_USE_CUDA
    rtk::CudaWarpedForwardProjectionImageFilter::Pointer forwardProjection;
    forwardProjection = rtk::CudaWarpedForwardProjectionImageFilter::New();
#else
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
#endif

  forwardProjection->SetInputProjectionStack( constantImageSource->GetOutput() );
  forwardProjection->SetInputVolume( reader->GetOutput() );
  forwardProjection->SetDisplacementField( dvfReader->GetOutput() );
  forwardProjection->SetGeometry( geometryReader->GetOutputObject() );
  projProbe.Start();
  
  if(!args_info.lowmem_flag)
    {
    TRY_AND_EXIT_ON_ITK_EXCEPTION( forwardProjection->Update() );
    }
  projProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << projProbe.GetMean() << ' ' << projProbe.GetUnit()
              << '.' << std::endl;

  // Write
  if(args_info.verbose_flag)
    std::cout << "Writing... " << std::flush;
  itk::TimeProbe writeProbe;
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( forwardProjection->GetOutput() );
  if(args_info.lowmem_flag)
    {
    writer->SetNumberOfStreamDivisions(sizeOutput[2]);
    }
  writeProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );
  writeProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << writeProbe.GetMean() << ' ' << projProbe.GetUnit()
              << '.' << std::endl;

  return EXIT_SUCCESS;
}