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

#include "rtkconjugategradient_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkNormalizedJosephBackProjectionImageFilter.h"

#ifdef RTK_USE_CUDA
  #include <itkCudaImage.h>
#endif
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkconjugategradient, args_info);

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
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkconjugategradient>(reader, args_info);

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
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkconjugategradient>(constantImageSource, args_info);

    // Copy output image's information from an existing file, if requested
    if (args_info.like_given)
      {
      typedef itk::ImageFileReader<  OutputImageType > LikeReaderType;
      LikeReaderType::Pointer likeReader = LikeReaderType::New();
      likeReader->SetFileName( args_info.like_arg );
      TRY_AND_EXIT_ON_ITK_EXCEPTION( likeReader->UpdateOutputInformation() );
      constantImageSource->SetInformationFromImage(likeReader->GetOutput());
      }

    inputFilter = constantImageSource;
    }

  // Read weights if given, otherwise perform unweighted least squares reconstruction
  typedef itk::ImageFileReader<  OutputImageType > WeightsReaderType;
  WeightsReaderType::Pointer weightsReader = WeightsReaderType::New();
  if(args_info.weights_given)
    {
    weightsReader->SetFileName( args_info.weights_arg );
    }

  // Set the forward and back projection filters to be used
  typedef rtk::ConjugateGradientConeBeamReconstructionFilter<OutputImageType> ConjugateGradientFilterType;
  ConjugateGradientFilterType::Pointer conjugategradient = ConjugateGradientFilterType::New();
  conjugategradient->SetForwardProjectionFilter(args_info.fp_arg);
  conjugategradient->SetBackProjectionFilter(args_info.bp_arg);

  conjugategradient->SetInput( inputFilter->GetOutput() );
  conjugategradient->SetInput(1, reader->GetOutput());
  if (args_info.weights_given)
    {
    conjugategradient->SetInput(2, weightsReader->GetOutput());
    conjugategradient->SetIsWeighted(true);
    }
  conjugategradient->SetGeometry( geometryReader->GetOutputObject() );
  conjugategradient->SetNumberOfIterations( args_info.niterations_arg );

  itk::TimeProbe readerProbe;
  if(args_info.time_flag)
    {
    std::cout << "Recording elapsed time... " << std::flush;
    readerProbe.Start();
    }

  TRY_AND_EXIT_ON_ITK_EXCEPTION( conjugategradient->Update() )

  if(args_info.time_flag)
    {
//    conjugategradient->PrintTiming(std::cout);
    readerProbe.Stop();
    std::cout << "It took...  " << readerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;
    }

  // Write
  typedef itk::ImageFileWriter< OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( conjugategradient->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
