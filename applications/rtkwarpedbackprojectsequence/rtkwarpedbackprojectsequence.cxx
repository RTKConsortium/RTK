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

#include "rtkwarpedbackprojectsequence_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkWarpProjectionStackToFourDImageFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkPhasesToInterpolationWeights.h"

#ifdef RTK_USE_CUDA
  #include "itkCudaImage.h"
#endif
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkwarpedbackprojectsequence, args_info);

  typedef float OutputPixelType;
  typedef itk::CovariantVector< OutputPixelType, 3 > DVFVectorType;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, 4 >  VolumeSeriesType;
  typedef itk::CudaImage< OutputPixelType, 3 >  ProjectionStackType;
  typedef itk::CudaImage<DVFVectorType, VolumeSeriesType::ImageDimension> DVFSequenceImageType;
  typedef itk::CudaImage<DVFVectorType, VolumeSeriesType::ImageDimension - 1> DVFImageType;
#else
  typedef itk::Image< OutputPixelType, 4 > VolumeSeriesType;
  typedef itk::Image< OutputPixelType, 3 > ProjectionStackType;
  typedef itk::Image<DVFVectorType, VolumeSeriesType::ImageDimension> DVFSequenceImageType;
  typedef itk::Image<DVFVectorType, VolumeSeriesType::ImageDimension - 1> DVFImageType;
#endif
  typedef ProjectionStackType                   VolumeType;
  typedef itk::ImageFileReader<  DVFSequenceImageType > DVFReaderType;

  // Projections reader
  typedef rtk::ProjectionsReader< ProjectionStackType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkwarpedbackprojectsequence>(reader, args_info);

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
    // Create new empty volume sequence
    typedef rtk::ConstantImageSource< VolumeSeriesType > ConstantImageSourceType;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkwarpedbackprojectsequence>(constantImageSource, args_info);

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

  // Read the phases file
  rtk::PhasesToInterpolationWeights::Pointer phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(args_info.signal_arg);
  phaseReader->SetNumberOfReconstructedFrames(inputFilter->GetOutput()->GetLargestPossibleRegion().GetSize(3));
  phaseReader->Update();  
  
  // Create the main filter, connect the basic inputs, and set the basic parameters
  typedef rtk::WarpProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType> ROOSTERFilterType;
  ROOSTERFilterType::Pointer rooster = ROOSTERFilterType::New();
  rooster->SetInputVolumeSeries(inputFilter->GetOutput() );
  rooster->SetInputProjectionStack(reader->GetOutput());
  rooster->SetGeometry( geometryReader->GetOutputObject() );
  rooster->SetWeights(phaseReader->GetOutput());
  rooster->SetCG_iterations( args_info.cgiter_arg );
  rooster->SetMainLoop_iterations( args_info.niter_arg );
  rooster->SetPhaseShift(args_info.shift_arg);
  rooster->SetCudaConjugateGradient(args_info.cudacg_flag);
  
  //Set the forward and back projection filters to be used
  rooster->SetForwardProjectionFilter(args_info.fp_arg);
  rooster->SetBackProjectionFilter(args_info.bp_arg);
  
  // For each optional regularization step, set whether or not
  // it should be performed, and provide the necessary inputs
  
  // Positivity
  if (args_info.nopositivity_flag)
    rooster->SetPerformPositivity(false);
  else
    rooster->SetPerformPositivity(true);
  
  // Motion mask
  typedef itk::ImageFileReader<  VolumeType > InputReaderType;
  if (args_info.motionmask_given)
    {
    InputReaderType::Pointer motionMaskReader = InputReaderType::New();
    motionMaskReader->SetFileName( args_info.motionmask_arg );
    motionMaskReader->Update();
    rooster->SetMotionMask(motionMaskReader->GetOutput());
    rooster->SetPerformMotionMask(true);
    }
  else
    rooster->SetPerformMotionMask(false);
    
  // Spatial TV
  if (args_info.gamma_space_given)
    {
    rooster->SetGammaTVSpace(args_info.gamma_space_arg);
    rooster->SetTV_iterations(args_info.tviter_arg);
    rooster->SetPerformTVSpatialDenoising(true);
    }
  else
    rooster->SetPerformTVSpatialDenoising(false);
  
  // Spatial wavelets
  if (args_info.threshold_given)
    {
    rooster->SetSoftThresholdWavelets(args_info.threshold_arg);
    rooster->SetOrder(args_info.order_arg);
    rooster->SetNumberOfLevels(args_info.levels_arg);
    rooster->SetPerformWaveletsSpatialDenoising(true);
    }
  else
    rooster->SetPerformWaveletsSpatialDenoising(false);
  
  // Temporal TV
  if (args_info.gamma_time_given)
    {
    rooster->SetGammaTVTime(args_info.gamma_time_arg);
    rooster->SetTV_iterations(args_info.tviter_arg);
    rooster->SetPerformTVTemporalDenoising(true);
    }
  else
    rooster->SetPerformTVTemporalDenoising(false);

  // Temporal L0
  if (args_info.lambda_time_arg)
    {
    rooster->SetLambdaL0Time(args_info.lambda_time_arg);
    rooster->SetL0_iterations(args_info.l0iter_arg);
    rooster->SetPerformL0TemporalDenoising(true);
    }
  else
    rooster->SetPerformL0TemporalDenoising(false);

  if (args_info.dvf_given)
    {
    rooster->SetPerformWarping(true);

    if(args_info.nn_flag)
      rooster->SetUseNearestNeighborInterpolationInWarping(true);

    // Read DVF
    DVFReaderType::Pointer dvfReader = DVFReaderType::New();
    dvfReader->SetFileName( args_info.dvf_arg );
    dvfReader->Update();
    rooster->SetDisplacementField(dvfReader->GetOutput());

    if (args_info.idvf_given)
      {
      rooster->SetComputeInverseWarpingByConjugateGradient(false);

      // Read inverse DVF if provided
      DVFReaderType::Pointer idvfReader = DVFReaderType::New();
      idvfReader->SetFileName( args_info.idvf_arg );
      idvfReader->Update();
      rooster->SetInverseDisplacementField(idvfReader->GetOutput());
      }
    }

  itk::TimeProbe readerProbe;
  if(args_info.time_flag)
    {
    std::cout << "Recording elapsed time... " << std::flush;
    readerProbe.Start();
    }

  TRY_AND_EXIT_ON_ITK_EXCEPTION( rooster->Update() );

  if(args_info.time_flag)
    {
    rooster->PrintTiming(std::cout);
    readerProbe.Stop();
    std::cout << "It took...  " << readerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;
    }

  // Write
  typedef itk::ImageFileWriter< VolumeSeriesType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( rooster->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
