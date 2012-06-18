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

#include "rtkfdk_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkProjectionsReader.h"
#include "rtkDisplacedDetectorImageFilter.h"
#include "rtkParkerShortScanImageFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#if CUDA_FOUND
# include "rtkCudaFDKConeBeamReconstructionFilter.h"
#endif
#if OPENCL_FOUND
# include "rtkOpenCLFDKConeBeamReconstructionFilter.h"
#endif

#include <itkRegularExpressionSeriesFileNames.h>
#include <itkStreamingImageFilter.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkfdk, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Generate file names
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(false);
  names->SetRegularExpression(args_info.regexp_arg);
  names->SetSubMatch(0);

  if(args_info.verbose_flag)
    std::cout << "Regular expression matches "
              << names->GetFileNames().size()
              << " file(s)..."
              << std::endl;

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->GenerateOutputInformation() );

  itk::TimeProbe readerProbe;
  if(!args_info.lowmem_flag)
    {
    if(args_info.verbose_flag)
      std::cout << "Reading... " << std::flush;
    readerProbe.Start();
    TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )
    readerProbe.Stop();
    if(args_info.verbose_flag)
      std::cout << "It took " << readerProbe.GetMeanTime() << ' ' << readerProbe.GetUnit() << std::endl;
    }

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

  // Displaced detector weighting
  typedef rtk::DisplacedDetectorImageFilter< OutputImageType > DDFType;
  DDFType::Pointer ddf = DDFType::New();
  ddf->SetInput( reader->GetOutput() );
  ddf->SetGeometry( geometryReader->GetOutputObject() );

  // Short scan image filter
  typedef rtk::ParkerShortScanImageFilter< OutputImageType > PSSFType;
  PSSFType::Pointer pssf = PSSFType::New();
  pssf->SetInput( ddf->GetOutput() );
  pssf->SetGeometry( geometryReader->GetOutputObject() );
  pssf->InPlaceOff();

  // Create reconstructed image
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkfdk>(constantImageSource, args_info);

  // This macro sets options for fdk filter which I can not see how to do better
  // because TFFTPrecision is not the same, e.g. for CPU and CUDA (SR)
#define SET_FELDKAMP_OPTIONS(f) \
    f->SetInput( 0, constantImageSource->GetOutput() ); \
    f->SetInput( 1, pssf->GetOutput() ); \
    f->SetGeometry( geometryReader->GetOutputObject() ); \
    f->GetRampFilter()->SetTruncationCorrection(args_info.pad_arg); \
    f->GetRampFilter()->SetHannCutFrequency(args_info.hann_arg);

  // FDK reconstruction filtering
  itk::ImageToImageFilter<OutputImageType, OutputImageType>::Pointer feldkamp;
  typedef rtk::FDKConeBeamReconstructionFilter< OutputImageType > FDKCPUType;
#if CUDA_FOUND
  typedef rtk::CudaFDKConeBeamReconstructionFilter                FDKCUDAType;
#endif
#if OPENCL_FOUND
  typedef rtk::OpenCLFDKConeBeamReconstructionFilter              FDKOPENCLType;
#endif
  if(!strcmp(args_info.hardware_arg, "cpu") )
    {
    feldkamp = FDKCPUType::New();
    SET_FELDKAMP_OPTIONS( static_cast<FDKCPUType*>(feldkamp.GetPointer()) );
    }
  else if(!strcmp(args_info.hardware_arg, "cuda") )
    {
#if CUDA_FOUND
    feldkamp = FDKCUDAType::New();
    SET_FELDKAMP_OPTIONS( static_cast<FDKCUDAType*>(feldkamp.GetPointer()) );
#else
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
#endif
    }
  else if(!strcmp(args_info.hardware_arg, "opencl") )
    {
#if OPENCL_FOUND
    feldkamp = FDKOPENCLType::New();
    SET_FELDKAMP_OPTIONS( static_cast<FDKOPENCLType*>(feldkamp.GetPointer()) );
#else
    std::cerr << "The program has not been compiled with opencl option" << std::endl;
    return EXIT_FAILURE;
#endif
    }


  // Streaming depending on streaming capability of writer
  typedef itk::StreamingImageFilter<OutputImageType, OutputImageType> StreamerType;
  StreamerType::Pointer streamerBP = StreamerType::New();
  streamerBP->SetInput( feldkamp->GetOutput() );
  streamerBP->SetNumberOfStreamDivisions( args_info.divisions_arg );

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( streamerBP->GetOutput() );

  if(args_info.verbose_flag)
    std::cout << "Reconstructing and writing... " << std::flush;
  itk::TimeProbe writerProbe;

  writerProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );
  writerProbe.Stop();

  if(args_info.verbose_flag)
    {
    std::cout << "It took " << writerProbe.GetMeanTime() << ' ' << readerProbe.GetUnit() << std::endl;
    if(!strcmp(args_info.hardware_arg, "cpu") )
      static_cast<FDKCPUType* >(feldkamp.GetPointer())->PrintTiming(std::cout);
#if CUDA_FOUND
    else if(!strcmp(args_info.hardware_arg, "cuda") )
      static_cast<FDKCUDAType*>(feldkamp.GetPointer())->PrintTiming(std::cout);
#endif
#if OPENCL_FOUND
    else if(!strcmp(args_info.hardware_arg, "opencl") )
      static_cast<FDKOPENCLType*>(feldkamp.GetPointer())->PrintTiming(std::cout);
#endif
    }

  return EXIT_SUCCESS;
}
