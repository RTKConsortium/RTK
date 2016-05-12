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

#include "rtkfourdconjugategradient_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkPhasesToInterpolationWeights.h"
#include "rtkDisplacedDetectorImageFilter.h"

#ifdef RTK_USE_CUDA
  #include "itkCudaImage.h"
  #include "rtkCudaConstantVolumeSeriesSource.h"
#endif

#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkfourdconjugategradient, args_info);

  typedef float OutputPixelType;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, 4 > VolumeSeriesType;
  typedef itk::CudaImage< OutputPixelType, 3 > ProjectionStackType;
#else
  typedef itk::Image< OutputPixelType, 4 > VolumeSeriesType;
  typedef itk::Image< OutputPixelType, 3 > ProjectionStackType;
#endif

  // Projections reader
  typedef rtk::ProjectionsReader< ProjectionStackType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkfourdconjugategradient>(reader, args_info);
  reader->UpdateLargestPossibleRegion();

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
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkfourdconjugategradient>(constantImageSource, args_info);

    // GenGetOpt can't handle default arguments for multiple arguments like dimension or spacing.
    // The only default it accepts is to set all components of a multiple argument to the same value.
    // Default dimension is 256^4, ie the number of reconstructed instants is 256. It has to be set to a more reasonable value
    // which is why a "frames" argument is introduced
    ConstantImageSourceType::SizeType inputSize = constantImageSource->GetSize();
    inputSize[3] = args_info.frames_arg;
    constantImageSource->SetSize(inputSize);

    inputFilter = constantImageSource;
    }
  inputFilter->Update();
  inputFilter->ReleaseDataFlagOn();

  // Read weights if given, otherwise default to weights all equal to one
  itk::ImageSource< ProjectionStackType >::Pointer weightsSource;
  if(args_info.weights_given)
    {
    typedef itk::ImageFileReader<  ProjectionStackType > WeightsReaderType;
    WeightsReaderType::Pointer weightsReader = WeightsReaderType::New();
    weightsReader->SetFileName( args_info.weights_arg );
    weightsSource = weightsReader;
    }
  else
    {
    typedef rtk::ConstantImageSource< ProjectionStackType > ConstantWeightsSourceType;
    ConstantWeightsSourceType::Pointer constantWeightsSource = ConstantWeightsSourceType::New();

    // Set the weights to be like the projections
    reader->UpdateOutputInformation();
    constantWeightsSource->SetInformationFromImage(reader->GetOutput());
    constantWeightsSource->SetConstant(1.0);
    weightsSource = constantWeightsSource;
    }

  // Apply the displaced detector weighting now, then re-order the projection weights
  // containing everything
  typedef rtk::DisplacedDetectorImageFilter<ProjectionStackType> DisplacedDetectorFilterType;
  DisplacedDetectorFilterType::Pointer displaced = DisplacedDetectorFilterType::New();
  displaced->SetInput(weightsSource->GetOutput());
  displaced->SetGeometry(geometryReader->GetOutputObject());
  displaced->SetPadOnTruncatedSide(false);
  displaced->Update();

  // Read the phases file
  rtk::PhasesToInterpolationWeights::Pointer phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(args_info.signal_arg);
  phaseReader->SetNumberOfReconstructedFrames(inputFilter->GetOutput()->GetLargestPossibleRegion().GetSize(3));
  phaseReader->Update();

  // Re-order geometry and projections
  // In the new order, projections with identical phases are packed together
  std::vector<double> signal = rtk::ReadSignalFile(args_info.signal_arg);
  std::vector<unsigned int> permutation = rtk::GetSortingPermutation(signal);

  // Create a new object for each object that has to be reordered
  // Geometry
  rtk::ThreeDCircularProjectionGeometry::Pointer orderedGeometry = rtk::ThreeDCircularProjectionGeometry::New();
  // Signal vector
  std::vector<double> orderedSignal;
  // Array of interpolation and splat weights
  itk::Array2D<float> orderedWeights = itk::Array2D<float> (phaseReader->GetOutput().rows(), phaseReader->GetOutput().cols());
  // Stack of projection
  ProjectionStackType::Pointer orderedProjs = ProjectionStackType::New();
  orderedProjs->CopyInformation(reader->GetOutput());
  orderedProjs->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  orderedProjs->Allocate();
  orderedProjs->FillBuffer(0);
  // Stack of projection weights
  ProjectionStackType::Pointer orderedProjectionWeights = ProjectionStackType::New();
  orderedProjectionWeights->CopyInformation(reader->GetOutput());
  orderedProjectionWeights->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
  orderedProjectionWeights->Allocate();
  orderedProjectionWeights->FillBuffer(0);

  // Declare regions used in the loop
  ProjectionStackType::RegionType unorderedRegion = orderedProjs->GetLargestPossibleRegion();
  ProjectionStackType::RegionType orderedRegion = orderedProjs->GetLargestPossibleRegion();
  unorderedRegion.SetSize(2, 1);
  orderedRegion.SetSize(2, 1);

  // Perform the copies
  for (unsigned int proj=0; proj<orderedProjs->GetLargestPossibleRegion().GetSize()[2]; proj++)
    {
    // Copy the projection data and the projection weights

    // Regions
    unorderedRegion.SetIndex(2, permutation[proj]);
    orderedRegion.SetIndex(2, proj);

    itk::ImageRegionIterator<ProjectionStackType> unorderedProjsIt(reader->GetOutput(), unorderedRegion);
    itk::ImageRegionIterator<ProjectionStackType> orderedProjsIt(orderedProjs, orderedRegion);

    itk::ImageRegionIterator<ProjectionStackType> unorderedWeightsIt(displaced->GetOutput(), unorderedRegion);
    itk::ImageRegionIterator<ProjectionStackType> orderedWeightsIt(orderedProjectionWeights, orderedRegion);

    // Actual copy
    while(!orderedProjsIt.IsAtEnd())
      {
      orderedProjsIt.Set(unorderedProjsIt.Get());
      orderedWeightsIt.Set(unorderedWeightsIt.Get());
      ++orderedWeightsIt;
      ++unorderedWeightsIt;
      ++orderedProjsIt;
      ++unorderedProjsIt;
      }

    // Copy the geometry
    orderedGeometry->AddProjectionInRadians(geometryReader->GetOutputObject()->GetSourceToIsocenterDistances()[permutation[proj]],
                                   geometryReader->GetOutputObject()->GetSourceToDetectorDistances()[permutation[proj]],
                                   geometryReader->GetOutputObject()->GetGantryAngles()[permutation[proj]],
                                   geometryReader->GetOutputObject()->GetProjectionOffsetsX()[permutation[proj]],
                                   geometryReader->GetOutputObject()->GetProjectionOffsetsY()[permutation[proj]],
                                   geometryReader->GetOutputObject()->GetOutOfPlaneAngles()[permutation[proj]],
                                   geometryReader->GetOutputObject()->GetInPlaneAngles()[permutation[proj]],
                                   geometryReader->GetOutputObject()->GetSourceOffsetsX()[permutation[proj]],
                                   geometryReader->GetOutputObject()->GetSourceOffsetsY()[permutation[proj]]);


    // Copy a column of the weights array
    for(unsigned int row=0; row<phaseReader->GetOutput().rows(); row++)
      orderedWeights[row][proj] = phaseReader->GetOutput()[row][permutation[proj]];

    // Copy the signal
    orderedSignal.push_back(signal[permutation[proj]]);
    }
  displaced->GetOutput()->ReleaseData();
  reader->GetOutput()->ReleaseData();
  weightsSource->GetOutput()->ReleaseData();

  // Set the forward and back projection filters to be used
  typedef rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType> ConjugateGradientFilterType;
  ConjugateGradientFilterType::Pointer conjugategradient = ConjugateGradientFilterType::New();
  conjugategradient->SetForwardProjectionFilter(args_info.fp_arg);
  conjugategradient->SetBackProjectionFilter(args_info.bp_arg);
  conjugategradient->SetInputVolumeSeries(inputFilter->GetOutput() );
  conjugategradient->SetNumberOfIterations( args_info.niterations_arg );
  conjugategradient->SetCudaConjugateGradient(args_info.cudacg_flag);

  // Set the newly ordered arguments
  conjugategradient->SetInputProjectionWeights(orderedProjectionWeights);
  conjugategradient->SetInputProjectionStack( orderedProjs );
  conjugategradient->SetGeometry( orderedGeometry );
  conjugategradient->SetWeights(orderedWeights);
  conjugategradient->SetSignal(orderedSignal);

  itk::TimeProbe readerProbe;
  if(args_info.time_flag)
    {
    std::cout << "Recording elapsed time... " << std::flush;
    readerProbe.Start();
    }

  TRY_AND_EXIT_ON_ITK_EXCEPTION( conjugategradient->Update() );

  if(args_info.time_flag)
    {
//    conjugategradient->PrintTiming(std::cout);
    readerProbe.Stop();
    std::cout << "It took...  " << readerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;
    }

  // Write
  typedef itk::ImageFileWriter< VolumeSeriesType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( conjugategradient->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
