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

#include "rtksart_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkSARTConeBeamReconstructionFilter.h"
#include "rtkPhaseGatingImageFilter.h"
#include "rtkIterationCommands.h"

#ifdef RTK_USE_CUDA
  #include "itkCudaImage.h"
#endif

#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtksart, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage< OutputPixelType, Dimension >;
#else
  using OutputImageType = itk::Image< OutputPixelType, Dimension >;
#endif

  // Projections reader
  using ReaderType = rtk::ProjectionsReader< OutputImageType >;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtksart>(reader, args_info);

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

  // Phase gating weights reader
  using PhaseGatingFilterType = rtk::PhaseGatingImageFilter<OutputImageType>;
  PhaseGatingFilterType::Pointer phaseGating = PhaseGatingFilterType::New();
  if (args_info.signal_given)
    {
    phaseGating->SetPhasesFileName(args_info.signal_arg);
    phaseGating->SetGatingWindowWidth(args_info.windowwidth_arg);
    phaseGating->SetGatingWindowCenter(args_info.windowcenter_arg);
    phaseGating->SetGatingWindowShape(args_info.windowshape_arg);
    phaseGating->SetInputProjectionStack(reader->GetOutput());
    phaseGating->SetInputGeometry(geometryReader->GetOutputObject());
    TRY_AND_EXIT_ON_ITK_EXCEPTION( phaseGating->Update() )
    }

  // Create input: either an existing volume read from a file or a blank image
  itk::ImageSource< OutputImageType >::Pointer inputFilter;
  if(args_info.input_given)
    {
    // Read an existing image to initialize the volume
    using InputReaderType = itk::ImageFileReader<  OutputImageType >;
    InputReaderType::Pointer inputReader = InputReaderType::New();
    inputReader->SetFileName( args_info.input_arg );
    inputFilter = inputReader;
    }
  else
    {
    // Create new empty volume
    using ConstantImageSourceType = rtk::ConstantImageSource< OutputImageType >;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtksart>(constantImageSource, args_info);
    inputFilter = constantImageSource;
    }

  // SART reconstruction filter
  rtk::SARTConeBeamReconstructionFilter< OutputImageType >::Pointer sart =
      rtk::SARTConeBeamReconstructionFilter< OutputImageType >::New();

  // Set the forward and back projection filters
  SetForwardProjectionFromGgo(args_info, sart.GetPointer());
  SetBackProjectionFromGgo(args_info, sart.GetPointer());
  sart->SetInput( inputFilter->GetOutput() );
  if (args_info.signal_given)
    {
    sart->SetInput(1, phaseGating->GetOutput());
    sart->SetGeometry( phaseGating->GetOutputGeometry() );
    sart->SetGatingWeights( phaseGating->GetGatingWeightsOnSelectedProjections() );
    }
  else
    {
    sart->SetInput(1, reader->GetOutput());
    sart->SetGeometry( geometryReader->GetOutputObject() );
    }
  sart->SetNumberOfIterations( args_info.niterations_arg );
  sart->SetNumberOfProjectionsPerSubset( args_info.nprojpersubset_arg );
  sart->SetLambda( args_info.lambda_arg );
  sart->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  if(args_info.positivity_flag)
    {
    sart->SetEnforcePositivity(true);
    }

  REPORT_ITERATIONS(sart, rtk::SARTConeBeamReconstructionFilter< OutputImageType >, OutputImageType)

  TRY_AND_EXIT_ON_ITK_EXCEPTION( sart->Update() )

  // Write
  using WriterType = itk::ImageFileWriter< OutputImageType >;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( sart->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
