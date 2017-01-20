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

#include "rtkiterativefdk_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkIterativeFDKConeBeamReconstructionFilter.h"
#ifdef RTK_USE_CUDA
#  include "rtkCudaIterativeFDKConeBeamReconstructionFilter.h"
#endif
#include <itkStreamingImageFilter.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkiterativefdk, args_info);

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
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkiterativefdk>(reader, args_info);

  itk::TimeProbe readerProbe;
  if(args_info.verbose_flag)
    std::cout << "Reading... " << std::flush;
  readerProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )
  readerProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << "It took " << readerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;

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

  // Create reconstructed image
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkiterativefdk>(constantImageSource, args_info);

  bool enforcePositivity;
  if(args_info.positivity_flag)
    enforcePositivity = true;
  else
    enforcePositivity = false;

  // Since the last template argument for IterativeFDKConeBeamReconstructionFilter is
  // double for the CPU version, and float for the CUDA one, we cannot have a single
  // pointer for both possibilities. In order to set the options only once,
  // we therefore create a macro, exactly as in rtkfdk.cxx
#define SET_IFDK_OPTIONS(f) \
  f->SetInput( 0, constantImageSource->GetOutput() ); \
  f->SetInput( 1, reader->GetOutput() ); \
  f->SetGeometry(geometryReader->GetOutputObject()); \
  f->SetForwardProjectionFilter(args_info.fp_arg); \
  f->SetNumberOfIterations(args_info.niterations_arg); \
  f->SetTruncationCorrection(args_info.pad_arg); \
  f->SetHannCutFrequency(args_info.hann_arg); \
  f->SetHannCutFrequencyY(args_info.hannY_arg); \
  f->SetProjectionSubsetSize(args_info.subsetsize_arg); \
  f->SetLambda( args_info.lambda_arg ); \
  f->SetEnforcePositivity( enforcePositivity );\
  f->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  // Create Iterative FDK filter and connect it
  typedef rtk::IterativeFDKConeBeamReconstructionFilter<OutputImageType, OutputImageType, double> IFDKCPUType;
  IFDKCPUType::Pointer ifdk;
#ifdef RTK_USE_CUDA
  typedef rtk::CudaIterativeFDKConeBeamReconstructionFilter IFDKCUDAType;
  IFDKCUDAType::Pointer ifdkCUDA;
#endif

  OutputImageType::Pointer IFDKOutputPointer;
  if(!strcmp(args_info.hardware_arg, "cpu") )
    {
    ifdk = IFDKCPUType::New();
    SET_IFDK_OPTIONS( ifdk );
    IFDKOutputPointer = ifdk->GetOutput();
    }
#ifdef RTK_USE_CUDA
  else if(!strcmp(args_info.hardware_arg, "cuda") )
    {
    ifdkCUDA = IFDKCUDAType::New();
    SET_IFDK_OPTIONS( ifdkCUDA );
    IFDKOutputPointer = ifdkCUDA->GetOutput();
    }
#endif

  // Write
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( IFDKOutputPointer );

  if(args_info.verbose_flag)
    std::cout << "Reconstructing and writing... " << std::flush;
  itk::TimeProbe writerProbe;

  writerProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )
  writerProbe.Stop();

  if(args_info.verbose_flag)
    {
    std::cout << "It took " << writerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;
    }

  return EXIT_SUCCESS;
}
