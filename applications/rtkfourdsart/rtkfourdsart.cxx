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

#include "rtkfourdsart_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkGeneralPurposeFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkFourDSARTConeBeamReconstructionFilter.h"
#include "rtkPhasesToInterpolationWeights.h"
#include "rtkDisplacedDetectorImageFilter.h"

#ifdef RTK_USE_CUDA
  #include "itkCudaImage.h"
  #include "rtkCudaConstantVolumeSeriesSource.h"
#endif
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkfourdsart, args_info);

  typedef float OutputPixelType;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, 4 >  VolumeSeriesType;
  typedef itk::CudaImage< OutputPixelType, 3 >  ProjectionStackType;
#else
  typedef itk::Image< OutputPixelType, 4 > VolumeSeriesType;
  typedef itk::Image< OutputPixelType, 3 > ProjectionStackType;
#endif
  typedef ProjectionStackType                   VolumeType;

  // Projections reader
  typedef rtk::ProjectionsReader< ProjectionStackType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkfourdsart>(reader, args_info);

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
  itk::ImageSource< VolumeSeriesType >::Pointer inputFilter;
  if(args_info.input_given)
    {
    // Read an existing image to initialize the volume
    typedef itk::ImageFileReader<  VolumeSeriesType > InputReaderType;
    InputReaderType::Pointer inputReader = InputReaderType::New();
    inputReader->SetFileName( args_info.input_arg );
    inputFilter = inputReader;
    }
  else
    {
    // Create new empty volume
    typedef rtk::ConstantImageSource< VolumeSeriesType > ConstantImageSourceType;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkfourdsart>(constantImageSource, args_info);

    // GenGetOpt can't handle default arguments for multiple arguments like dimension or spacing.
    // The only default it accepts is to set all components of a multiple argument to the same value.
    // Default dimension is 256^4, ie the number of reconstructed instants is 256. It has to be set to a more reasonable value
    // which is why a "frames" argument is introduced
    ConstantImageSourceType::SizeType inputSize = constantImageSource->GetSize();
    inputSize[3] = args_info.frames_arg;
    constantImageSource->SetSize(inputSize);

    inputFilter = constantImageSource;
    }
  TRY_AND_EXIT_ON_ITK_EXCEPTION( inputFilter->Update() )
  inputFilter->ReleaseDataFlagOn();

  // Read the phases file
  rtk::PhasesToInterpolationWeights::Pointer phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(args_info.signal_arg);
  phaseReader->SetNumberOfReconstructedFrames(inputFilter->GetOutput()->GetLargestPossibleRegion().GetSize(3));
  TRY_AND_EXIT_ON_ITK_EXCEPTION( phaseReader->Update() )
  
  // 4D SART reconstruction filter
  rtk::FourDSARTConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType >::Pointer fourdsart =
      rtk::FourDSARTConeBeamReconstructionFilter< VolumeSeriesType, ProjectionStackType >::New();

  // Set the forward and back projection filters
  fourdsart->SetForwardProjectionFilter(args_info.fp_arg);
  fourdsart->SetBackProjectionFilter(args_info.bp_arg);
  fourdsart->SetInputVolumeSeries(inputFilter->GetOutput() );
  fourdsart->SetInputProjectionStack(reader->GetOutput());
  fourdsart->SetGeometry( geometryReader->GetOutputObject() );
  fourdsart->SetNumberOfIterations( args_info.niterations_arg );
  fourdsart->SetNumberOfProjectionsPerSubset( args_info.nprojpersubset_arg );
  fourdsart->SetWeights(phaseReader->GetOutput());
  fourdsart->SetSignal(rtk::ReadSignalFile(args_info.signal_arg));
  fourdsart->SetLambda( args_info.lambda_arg );
  fourdsart->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  itk::TimeProbe totalTimeProbe;
  if(args_info.time_flag)
    {
    std::cout << "Recording elapsed time... " << std::endl << std::flush;
    totalTimeProbe.Start();
    }
  if(args_info.positivity_flag)
    {
    fourdsart->SetEnforcePositivity(true);
    }

  TRY_AND_EXIT_ON_ITK_EXCEPTION( fourdsart->Update() )

  if(args_info.time_flag)
    {
    fourdsart->PrintTiming(std::cout);
    totalTimeProbe.Stop();
    std::cout << "It took...  " << totalTimeProbe.GetMean() << ' ' << totalTimeProbe.GetUnit() << std::endl;
    }

  // Write
  typedef itk::ImageFileWriter< VolumeSeriesType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( fourdsart->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
