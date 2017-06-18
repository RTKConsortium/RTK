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

#include "rtkmcrooster_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkGeneralPurposeFunctions.h"

#include "rtkMotionCompensatedFourDROOSTERConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkSignalToInterpolationWeights.h"
#include "rtkReorderProjectionsImageFilter.h"
#include "rtkWarpSequenceImageFilter.h"

#ifdef RTK_USE_CUDA
  #include "itkCudaImage.h"
#endif
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkmcrooster, args_info);

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
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkmcrooster>(reader, args_info);

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
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkmcrooster>(constantImageSource, args_info);

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

  // Re-order geometry and projections
  // In the new order, projections with identical phases are packed together
  std::vector<double> signal = rtk::ReadSignalFile(args_info.signal_arg);
  typedef rtk::ReorderProjectionsImageFilter<ProjectionStackType> ReorderProjectionsFilterType;
  ReorderProjectionsFilterType::Pointer reorder = ReorderProjectionsFilterType::New();
  reorder->SetInput(reader->GetOutput());
  reorder->SetInputGeometry(geometryReader->GetOutputObject());
  reorder->SetInputSignal(signal);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reorder->Update() )

  // Release the memory holding the stack of original projections
  reader->GetOutput()->ReleaseData();

  // Compute the interpolation weights
  rtk::SignalToInterpolationWeights::Pointer signalToInterpolationWeights = rtk::SignalToInterpolationWeights::New();
  signalToInterpolationWeights->SetSignal(reorder->GetOutputSignal());
  signalToInterpolationWeights->SetNumberOfReconstructedFrames(inputFilter->GetOutput()->GetLargestPossibleRegion().GetSize(3));
  TRY_AND_EXIT_ON_ITK_EXCEPTION( signalToInterpolationWeights->Update() )

  // Create the 4DROOSTER filter, connect the basic inputs, and set the basic parameters
  typedef rtk::MotionCompensatedFourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType> MCROOSTERFilterType;
  MCROOSTERFilterType::Pointer mcrooster = MCROOSTERFilterType::New();
  mcrooster->SetForwardProjectionFilter(args_info.fp_arg);
  mcrooster->SetBackProjectionFilter(args_info.bp_arg);
  mcrooster->SetInputVolumeSeries(inputFilter->GetOutput() );
  mcrooster->SetCG_iterations( args_info.cgiter_arg );
  mcrooster->SetMainLoop_iterations( args_info.niter_arg );
  mcrooster->SetCudaConjugateGradient(args_info.cudacg_flag);
  mcrooster->SetUseCudaCyclicDeformation(args_info.cudadvfinterpolation_flag);
  mcrooster->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  // Set the newly ordered arguments
  mcrooster->SetInputProjectionStack( reorder->GetOutput() );
  mcrooster->SetGeometry( reorder->GetOutputGeometry() );
  mcrooster->SetWeights(signalToInterpolationWeights->GetOutput());
  mcrooster->SetSignal(reorder->GetOutputSignal());

  // For each optional regularization step, set whether or not
  // it should be performed, and provide the necessary inputs
  
  // Positivity
  if (args_info.nopositivity_flag)
    mcrooster->SetPerformPositivity(false);
  else
    mcrooster->SetPerformPositivity(true);
  
  // Motion mask
  typedef itk::ImageFileReader<  VolumeType > InputReaderType;
  if (args_info.motionmask_given)
    {
    InputReaderType::Pointer motionMaskReader = InputReaderType::New();
    motionMaskReader->SetFileName( args_info.motionmask_arg );
    TRY_AND_EXIT_ON_ITK_EXCEPTION( motionMaskReader->Update() )
    mcrooster->SetMotionMask(motionMaskReader->GetOutput());
    mcrooster->SetPerformMotionMask(true);
    }
  else
    mcrooster->SetPerformMotionMask(false);
    
  // Spatial TV
  if (args_info.gamma_space_given)
    {
    mcrooster->SetGammaTVSpace(args_info.gamma_space_arg);
    mcrooster->SetTV_iterations(args_info.tviter_arg);
    mcrooster->SetPerformTVSpatialDenoising(true);
    }
  else
    mcrooster->SetPerformTVSpatialDenoising(false);
  
  // Spatial wavelets
  if (args_info.threshold_given)
    {
    mcrooster->SetSoftThresholdWavelets(args_info.threshold_arg);
    mcrooster->SetOrder(args_info.order_arg);
    mcrooster->SetNumberOfLevels(args_info.levels_arg);
    mcrooster->SetPerformWaveletsSpatialDenoising(true);
    }
  else
    mcrooster->SetPerformWaveletsSpatialDenoising(false);
  
  // Temporal TV
  if (args_info.gamma_time_given)
    {
    mcrooster->SetGammaTVTime(args_info.gamma_time_arg);
    mcrooster->SetTV_iterations(args_info.tviter_arg);
    mcrooster->SetPerformTVTemporalDenoising(true);
    }
  else
    mcrooster->SetPerformTVTemporalDenoising(false);

  // Temporal L0
  if (args_info.lambda_time_arg)
    {
    mcrooster->SetLambdaL0Time(args_info.lambda_time_arg);
    mcrooster->SetL0_iterations(args_info.l0iter_arg);
    mcrooster->SetPerformL0TemporalDenoising(true);
    }
  else
    mcrooster->SetPerformL0TemporalDenoising(false);

  // Read DVF
  DVFReaderType::Pointer dvfReader = DVFReaderType::New();
  dvfReader->SetFileName( args_info.dvf_arg );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dvfReader->Update() )
  mcrooster->SetDisplacementField(dvfReader->GetOutput());

  // Read inverse DVF if provided
  DVFReaderType::Pointer idvfReader = DVFReaderType::New();
  idvfReader->SetFileName( args_info.idvf_arg );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( idvfReader->Update() )
  mcrooster->SetInverseDisplacementField(idvfReader->GetOutput());

  itk::TimeProbe readerProbe;
  if(args_info.time_flag)
    {
    std::cout << "Recording elapsed time... " << std::flush;
    readerProbe.Start();
    }

  TRY_AND_EXIT_ON_ITK_EXCEPTION( mcrooster->Update() )

//  if(args_info.time_flag)
//    {
//    mcrooster->PrintTiming(std::cout);
//    readerProbe.Stop();
//    std::cout << "It took...  " << readerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;
//    }

  typedef itk::ImageFileWriter< VolumeSeriesType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );

  typedef rtk::WarpSequenceImageFilter<VolumeSeriesType, DVFSequenceImageType, ProjectionStackType, DVFImageType> WarpSequenceFilterType;
  WarpSequenceFilterType::Pointer warp = WarpSequenceFilterType::New();

  if(args_info.nofinalwarp_flag)
    {
    // MCROOSTER outputs a motion-compensated reconstruction
    writer->SetInput( mcrooster->GetOutput() );
    TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )
    }
  else
    {
    // Warp the output of MCROOSTER with the inverse field so
    // that it is similar to that of rtkfourdrooster
    warp->SetInput(mcrooster->GetOutput());
    warp->SetDisplacementField(idvfReader->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION( warp->Update() )

    // Write
    writer->SetInput( warp->GetOutput() );
    TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )
    }

  return EXIT_SUCCESS;
}
