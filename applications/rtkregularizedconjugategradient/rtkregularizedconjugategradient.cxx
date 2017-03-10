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

#include "rtkregularizedconjugategradient_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkRegularizedConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

#ifdef RTK_USE_CUDA
  #include "itkCudaImage.h"
  #include "rtkCudaConstantVolumeSource.h"
#endif
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkregularizedconjugategradient, args_info);

  typedef float OutputPixelType;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, 3 > OutputImageType;
#else
  typedef itk::Image< OutputPixelType, 3 > OutputImageType;
#endif

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkregularizedconjugategradient>(reader, args_info);

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
  itk::ImageSource< OutputImageType >::Pointer inputFilter;
  if(args_info.input_given)
    {
    // Read an existing image to initialize the volume
    typedef itk::ImageFileReader<  OutputImageType > InputReaderType;
    InputReaderType::Pointer inputReader = InputReaderType::New();
    inputReader->SetFileName( args_info.input_arg );
    inputFilter = inputReader;
    }
  else
    {
    // Create new empty volume
    typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkregularizedconjugategradient>(constantImageSource, args_info);
    inputFilter = constantImageSource;
    }
  TRY_AND_EXIT_ON_ITK_EXCEPTION( inputFilter->Update() )
  inputFilter->ReleaseDataFlagOn();

  // Read weights if given, otherwise default to weights all equal to one
  itk::ImageSource< OutputImageType >::Pointer weightsSource;
  if(args_info.weights_given)
    {
    typedef itk::ImageFileReader<  OutputImageType > WeightsReaderType;
    WeightsReaderType::Pointer weightsReader = WeightsReaderType::New();
    weightsReader->SetFileName( args_info.weights_arg );
    weightsSource = weightsReader;
    }
  else
    {
    typedef rtk::ConstantImageSource< OutputImageType > ConstantWeightsSourceType;
    ConstantWeightsSourceType::Pointer constantWeightsSource = ConstantWeightsSourceType::New();

    // Set the weights to be like the projections
    TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->UpdateOutputInformation() )
    constantWeightsSource->SetInformationFromImage(reader->GetOutput());
    constantWeightsSource->SetConstant(1.0);
    weightsSource = constantWeightsSource;
    }

  // Read Support Mask if given
  itk::ImageSource< OutputImageType >::Pointer supportmaskSource;
  if(args_info.mask_given)
    {
    typedef itk::ImageFileReader<  OutputImageType > MaskReaderType;
    MaskReaderType::Pointer supportmaskReader = MaskReaderType::New();
    supportmaskReader->SetFileName( args_info.mask_arg );
    supportmaskSource = supportmaskReader;
    }

  // Set the forward and back projection filters to be used
  typedef rtk::RegularizedConjugateGradientConeBeamReconstructionFilter<OutputImageType> ConjugateGradientFilterType;
  ConjugateGradientFilterType::Pointer regularizedConjugateGradient = ConjugateGradientFilterType::New();
  regularizedConjugateGradient->SetForwardProjectionFilter(args_info.fp_arg);
  regularizedConjugateGradient->SetBackProjectionFilter(args_info.bp_arg);
  regularizedConjugateGradient->SetInputVolume(inputFilter->GetOutput() );
  regularizedConjugateGradient->SetInputProjectionStack(reader->GetOutput());
  regularizedConjugateGradient->SetInputWeights( weightsSource->GetOutput());
  regularizedConjugateGradient->SetGeometry( geometryReader->GetOutputObject() );
  regularizedConjugateGradient->SetMainLoop_iterations( args_info.niter_arg );
  regularizedConjugateGradient->SetCudaConjugateGradient(!args_info.nocudacg_flag);
  regularizedConjugateGradient->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);
  if(args_info.mask_given)
    {
    regularizedConjugateGradient->SetSupportMask(supportmaskSource->GetOutput() );
    }
  regularizedConjugateGradient->SetIterationCosts(args_info.costs_flag);

  // Positivity
  if (args_info.nopositivity_flag)
    regularizedConjugateGradient->SetPerformPositivity(false);
  else
    regularizedConjugateGradient->SetPerformPositivity(true);

  if (args_info.gammalaplacian_given)
    {
    regularizedConjugateGradient->SetRegularizedCG(true);
    regularizedConjugateGradient->SetGamma(args_info.gammalaplacian_arg);
    }
  else
    regularizedConjugateGradient->SetRegularizedCG(false);

  // Spatial TV
  if (args_info.gammatv_given)
    {
    regularizedConjugateGradient->SetGammaTV(args_info.gammatv_arg);
    regularizedConjugateGradient->SetTV_iterations(args_info.tviter_arg);
    regularizedConjugateGradient->SetPerformTVSpatialDenoising(true);
    }
  else
    regularizedConjugateGradient->SetPerformTVSpatialDenoising(false);

  // Spatial wavelets
  if (args_info.threshold_given)
    {
    regularizedConjugateGradient->SetSoftThresholdWavelets(args_info.threshold_arg);
    regularizedConjugateGradient->SetOrder(args_info.order_arg);
    regularizedConjugateGradient->SetNumberOfLevels(args_info.levels_arg);
    regularizedConjugateGradient->SetPerformWaveletsSpatialDenoising(true);
    }
  else
    regularizedConjugateGradient->SetPerformWaveletsSpatialDenoising(false);

  // Sparsity in image domain
  if (args_info.soft_given)
    {
    regularizedConjugateGradient->SetSoftThresholdOnImage(args_info.soft_arg);
    regularizedConjugateGradient->SetPerformSoftThresholdOnImage(true);
    }
  else
    regularizedConjugateGradient->SetSoftThresholdOnImage(false);

  itk::TimeProbe readerProbe;
  if(args_info.time_flag)
    {
    std::cout << "Recording elapsed time... " << std::flush;
    readerProbe.Start();
    }

  TRY_AND_EXIT_ON_ITK_EXCEPTION( regularizedConjugateGradient->Update() )

  if(args_info.time_flag)
    {
    regularizedConjugateGradient->PrintTiming(std::cout);
    readerProbe.Stop();
    std::cout << "It took...  " << readerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;
    }

  // Write
  typedef itk::ImageFileWriter< OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( regularizedConjugateGradient->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
