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

#include "rtkfourdfdk_ggo.h"
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
#include "rtkSelectOneProjectionPerCycleImageFilter.h"

#include <itkExtractImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkStreamingImageFilter.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkfourdfdk, args_info);

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
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkfourdfdk>(reader, args_info);

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

  // Part specific to 4D
  typedef rtk::SelectOneProjectionPerCycleImageFilter<OutputImageType> SelectorType;
  SelectorType::Pointer selector = SelectorType::New();
  selector->SetInput( reader->GetOutput() );
  selector->SetInputGeometry( geometryReader->GetOutputObject() );
  selector->SetSignalFilename( args_info.signal_arg );

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
  ddf->SetInput( selector->GetOutput() );
  ddf->SetGeometry( selector->GetOutputGeometry() );

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
  pssf->SetGeometry( selector->GetOutputGeometry() );
  pssf->InPlaceOff();

  // Create one frame of the reconstructed image
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkfourdfdk>(constantImageSource, args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantImageSource->Update())

  // This macro sets options for fdk filter which I can not see how to do better
  // because TFFTPrecision is not the same, e.g. for CPU and CUDA (SR)
#define SET_FELDKAMP_OPTIONS(f) \
    f->SetInput( 0, constantImageSource->GetOutput() ); \
    f->SetInput( 1, pssf->GetOutput() ); \
    f->SetGeometry( selector->GetOutputGeometry() ); \
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

    pfeldkamp = feldkamp->GetOutput();
    }
#ifdef RTK_USE_CUDA
  else if(!strcmp(args_info.hardware_arg, "cuda") )
    {
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
  TRY_AND_EXIT_ON_ITK_EXCEPTION( streamerBP->UpdateOutputInformation() )

  // Create empty 4D image
  typedef itk::Image< OutputPixelType, Dimension+1 >       FourDOutputImageType;
  typedef rtk::ConstantImageSource< FourDOutputImageType > FourDConstantImageSourceType;
  FourDConstantImageSourceType::Pointer fourDConstantImageSource = FourDConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<FourDConstantImageSourceType, args_info_rtkfourdfdk>(fourDConstantImageSource, args_info);

  // GenGetOpt can't handle default arguments for multiple arguments like dimension or spacing.
  // The only default it accepts is to set all components of a multiple argument to the same value.
  // Default dimension is 256^4, ie the number of reconstructed instants is 256. It has to be set to a more reasonable value
  // which is why a "frames" argument is introduced
  FourDConstantImageSourceType::SizeType fourDInputSize(fourDConstantImageSource->GetSize());
  fourDInputSize[3] = args_info.frames_arg;
  fourDConstantImageSource->SetSize(fourDInputSize);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(fourDConstantImageSource->Update())

  // Go over each frame, reconstruct 3D frame and paste with iterators in 4D image
  for(int f=0; f<args_info.frames_arg; f++)
    {
    if(args_info.verbose_flag)
      std::cout << "Reconstructing frame #"
                << f
                << "..."
                << std::endl;
    selector->SetPhase(f/(double)args_info.frames_arg);
    TRY_AND_EXIT_ON_ITK_EXCEPTION( streamerBP->UpdateLargestPossibleRegion() )

    FourDConstantImageSourceType::OutputImageRegionType region;
    region = fourDConstantImageSource->GetOutput()->GetLargestPossibleRegion();
    region.SetIndex(3, f);
    region.SetSize(3, 1);

    itk::ImageRegionIterator<FourDOutputImageType> it4D(fourDConstantImageSource->GetOutput(),
                                                        region);
    itk::ImageRegionIterator<CPUOutputImageType>   it3D(streamerBP->GetOutput(),
                                                        streamerBP->GetOutput()->GetLargestPossibleRegion() );
    while(!it3D.IsAtEnd())
      {
      it4D.Set(it3D.Get());
      ++it4D;
      ++it3D;
      }
    }

  // Write
  typedef itk::ImageFileWriter<FourDOutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( fourDConstantImageSource->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
