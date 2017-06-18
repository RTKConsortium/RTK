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
#include "rtkDisplacedDetectorForOffsetFieldOfViewImageFilter.h"
#include "rtkParkerShortScanImageFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#ifdef RTK_USE_CUDA
#  include "rtkCudaDisplacedDetectorImageFilter.h"
//TODO #  include "rtkCudaDisplacedDetectorForOffsetFieldOfViewImageFilter.h"
#  include "rtkCudaParkerShortScanImageFilter.h"
#  include "rtkCudaFDKConeBeamReconstructionFilter.h"
#endif
#include "rtkFDKWarpBackProjectionImageFilter.h"
#include "rtkCyclicDeformationImageFilter.h"

#include <itkStreamingImageFilter.h>
#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  #include <itkImageRegionSplitterDirection.h>
#endif
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkfdk, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension >     CPUOutputImageType;
#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef CPUOutputImageType                           OutputImageType;
#endif

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkfdk>(reader, args_info);

  itk::TimeProbe readerProbe;
  if(!args_info.lowmem_flag)
    {
    if(args_info.verbose_flag)
      std::cout << "Reading... " << std::flush;
    readerProbe.Start();
    TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )
    readerProbe.Stop();
    if(args_info.verbose_flag)
      std::cout << "It took " << readerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;
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

  // Check on hardware parameter
#ifndef RTK_USE_CUDA
  if(!strcmp(args_info.hardware_arg, "cuda") )
    {
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
    }
#endif

  // Displaced detector weighting
  typedef rtk::DisplacedDetectorImageFilter< OutputImageType >                     DDFCPUType;
  typedef rtk::DisplacedDetectorForOffsetFieldOfViewImageFilter< OutputImageType > DDFOFFFOVType;
#ifdef RTK_USE_CUDA
  typedef rtk::CudaDisplacedDetectorImageFilter DDFType;
#else
  typedef rtk::DisplacedDetectorForOffsetFieldOfViewImageFilter< OutputImageType > DDFType;
#endif
  DDFCPUType::Pointer ddf;
  if(!strcmp(args_info.hardware_arg, "cuda") )
    ddf = DDFType::New();
  else
    ddf = DDFOFFFOVType::New();
  ddf->SetInput( reader->GetOutput() );
  ddf->SetGeometry( geometryReader->GetOutputObject() );
  ddf->SetDisable(args_info.nodisplaced_flag);

  // Short scan image filter
  typedef rtk::ParkerShortScanImageFilter< OutputImageType > PSSFCPUType;
#ifdef RTK_USE_CUDA
  typedef rtk::CudaParkerShortScanImageFilter PSSFType;
#else
  typedef rtk::ParkerShortScanImageFilter< OutputImageType > PSSFType;
#endif
  PSSFCPUType::Pointer pssf;
  if(!strcmp(args_info.hardware_arg, "cuda") )
    pssf = PSSFType::New();
  else
    pssf = PSSFCPUType::New();
  pssf->SetInput( ddf->GetOutput() );
  pssf->SetGeometry( geometryReader->GetOutputObject() );
  pssf->InPlaceOff();

  // Create reconstructed image
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkfdk>(constantImageSource, args_info);

  // Motion-compensated objects for the compensation of a cyclic deformation.
  // Although these will only be used if the command line options for motion
  // compensation are set, we still create the object before hand to avoid auto
  // destruction.
  typedef itk::Vector<float,3> DVFPixelType;
  typedef itk::Image< DVFPixelType, 3 > DVFImageType;
  typedef rtk::CyclicDeformationImageFilter< DVFImageType > DeformationType;
  typedef itk::ImageFileReader<DeformationType::InputImageType> DVFReaderType;
  DVFReaderType::Pointer dvfReader = DVFReaderType::New();
  DeformationType::Pointer def = DeformationType::New();
  def->SetInput(dvfReader->GetOutput());
  typedef rtk::FDKWarpBackProjectionImageFilter<OutputImageType, OutputImageType, DeformationType> WarpBPType;
  WarpBPType::Pointer bp = WarpBPType::New();
  bp->SetDeformation(def);
  bp->SetGeometry( geometryReader->GetOutputObject() );

  // This macro sets options for fdk filter which I can not see how to do better
  // because TFFTPrecision is not the same, e.g. for CPU and CUDA (SR)
#define SET_FELDKAMP_OPTIONS(f) \
    f->SetInput( 0, constantImageSource->GetOutput() ); \
    f->SetInput( 1, pssf->GetOutput() ); \
    f->SetGeometry( geometryReader->GetOutputObject() ); \
    f->GetRampFilter()->SetTruncationCorrection(args_info.pad_arg); \
    f->GetRampFilter()->SetHannCutFrequency(args_info.hann_arg); \
    f->GetRampFilter()->SetHannCutFrequencyY(args_info.hannY_arg); \
    f->SetProjectionSubsetSize(args_info.subsetsize_arg)

  // FDK reconstruction filtering
  typedef rtk::FDKConeBeamReconstructionFilter< OutputImageType > FDKCPUType;
  FDKCPUType::Pointer feldkamp;
#ifdef RTK_USE_CUDA
  typedef rtk::CudaFDKConeBeamReconstructionFilter FDKCUDAType;
  FDKCUDAType::Pointer feldkampCUDA;
#endif
  itk::Image< OutputPixelType, Dimension > *pfeldkamp = ITK_NULLPTR;
  if(!strcmp(args_info.hardware_arg, "cpu") )
    {
    feldkamp = FDKCPUType::New();
    SET_FELDKAMP_OPTIONS( feldkamp );

    // Motion compensated CBCT settings
    if(args_info.signal_given && args_info.dvf_given)
      {
      dvfReader->SetFileName(args_info.dvf_arg);
      def->SetSignalFilename(args_info.signal_arg);
      feldkamp->SetBackProjectionFilter( bp.GetPointer() );
      }
    pfeldkamp = feldkamp->GetOutput();
    }
#ifdef RTK_USE_CUDA
  else if(!strcmp(args_info.hardware_arg, "cuda") )
    {
    // Motion compensation not supported in cuda
    if(args_info.signal_given && args_info.dvf_given)
      {
      std::cerr << "Motion compensation is not supported in CUDA. Aborting" << std::endl;
      return EXIT_FAILURE;
      }

    feldkampCUDA = FDKCUDAType::New();
    SET_FELDKAMP_OPTIONS( feldkampCUDA );
    pfeldkamp = feldkampCUDA->GetOutput();
    }
#endif

  // Streaming depending on streaming capability of writer
  typedef itk::StreamingImageFilter<CPUOutputImageType, CPUOutputImageType> StreamerType;
  StreamerType::Pointer streamerBP = StreamerType::New();
  streamerBP->SetInput( pfeldkamp );
  streamerBP->SetNumberOfStreamDivisions( args_info.divisions_arg );
#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  itk::ImageRegionSplitterDirection::Pointer splitter = itk::ImageRegionSplitterDirection::New();
  splitter->SetDirection(2); // Prevent splitting along z axis. As a result, splitting will be performed along y axis
  streamerBP->SetRegionSplitter(splitter);
#endif

  // Write
  typedef itk::ImageFileWriter<CPUOutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( streamerBP->GetOutput() );

  if(args_info.verbose_flag)
    std::cout << "Reconstructing and writing... " << std::flush;
  itk::TimeProbe writerProbe;

  writerProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )
  writerProbe.Stop();

  if(args_info.verbose_flag)
    {
    std::cout << "It took " << writerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;
    if(!strcmp(args_info.hardware_arg, "cpu") )
      feldkamp->PrintTiming(std::cout);
#ifdef RTK_USE_CUDA
    else if(!strcmp(args_info.hardware_arg, "cuda") )
      feldkampCUDA->PrintTiming(std::cout);
#endif
    std::cout << std::endl;
    }

  return EXIT_SUCCESS;
}
