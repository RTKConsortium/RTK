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

#include "rtkvectorconjugategradient_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkConjugateGradientConeBeamReconstructionFilter.h"

#include <iostream>
#include <fstream>
#include <iterator>

#ifdef RTK_USE_CUDA
  #include <itkCudaImage.h>
#endif
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkvectorconjugategradient, args_info);

  const unsigned int Dimension = 3;
  const unsigned int nMaterials = 3;

  typedef float DataType;
  typedef itk::Vector<DataType, nMaterials> PixelType;
  typedef itk::Vector<DataType, nMaterials * nMaterials> WeightsType;

  std::vector<double> costs;
  std::ostream_iterator<double> costs_it(std::cout << std::setprecision(15),"\n");

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< DataType, Dimension > SingleComponentImageType;
  typedef itk::CudaImage< PixelType, Dimension > OutputImageType;
  typedef itk::CudaImage< WeightsType, Dimension > WeightsImageType;
#else
  typedef itk::Image< DataType, Dimension > SingleComponentImageType;
  typedef itk::Image< PixelType, Dimension > OutputImageType;
  typedef itk::Image< WeightsType, Dimension > WeightsImageType;
#endif

  // Projections reader
  typedef itk::ImageFileReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(args_info.projections_arg);
  reader->Update();

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
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkvectorconjugategradient>(constantImageSource, args_info);
    inputFilter = constantImageSource;
    }
  inputFilter->Update();

  // Read weights if given, otherwise default to weights all equal to one
  itk::ImageSource< WeightsImageType >::Pointer weightsSource;
  if(args_info.weights_given)
    {
    typedef itk::ImageFileReader<  WeightsImageType > WeightsReaderType;
    WeightsReaderType::Pointer weightsReader = WeightsReaderType::New();
    weightsReader->SetFileName( args_info.weights_arg );
    weightsSource = weightsReader;
    }
  else
    {
    typedef rtk::ConstantImageSource< WeightsImageType > ConstantWeightsSourceType;
    ConstantWeightsSourceType::Pointer constantWeightsSource = ConstantWeightsSourceType::New();
    
    // Set the weights to the identity matrix
    TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->UpdateOutputInformation() )
    constantWeightsSource->SetInformationFromImage(reader->GetOutput());
    WeightsType constantWeight = itk::NumericTraits<WeightsType>::ZeroValue(constantWeight);
    for (unsigned int i=0; i< nMaterials; i++)
      constantWeight[i + i*nMaterials] = 1;

    constantWeightsSource->SetConstant(constantWeight);
    weightsSource = constantWeightsSource;
    }

  // Read Support Mask if given
  itk::ImageSource< SingleComponentImageType >::Pointer supportmaskSource;
  if(args_info.mask_given)
    {
    typedef itk::ImageFileReader<  SingleComponentImageType > MaskReaderType;
    MaskReaderType::Pointer supportmaskReader = MaskReaderType::New();
    supportmaskReader->SetFileName( args_info.mask_arg );
    supportmaskSource = supportmaskReader;
    }

  // Set the forward and back projection filters to be used
  typedef rtk::ConjugateGradientConeBeamReconstructionFilter<OutputImageType, SingleComponentImageType, WeightsImageType> ConjugateGradientFilterType;
  ConjugateGradientFilterType::Pointer conjugategradient = ConjugateGradientFilterType::New();
//  conjugategradient->SetForwardProjectionFilter(ConjugateGradientFilterType::FP_JOSEPH);
//  conjugategradient->SetBackProjectionFilter(ConjugateGradientFilterType::BP_JOSEPH);
  SetForwardProjectionFromGgo(args_info, conjugategradient.GetPointer());
  SetBackProjectionFromGgo(args_info, conjugategradient.GetPointer());
  conjugategradient->SetInputVolume( inputFilter->GetOutput() );
  conjugategradient->SetInputProjectionStack( reader->GetOutput());
  conjugategradient->SetInputWeights( weightsSource->GetOutput());
  conjugategradient->SetCudaConjugateGradient(!args_info.nocudacg_flag);
  if(args_info.mask_given)
    {
    conjugategradient->SetSupportMask(supportmaskSource->GetOutput() );
    }
  conjugategradient->SetIterationCosts(args_info.costs_flag);

  if (args_info.tikhonov_given)
    conjugategradient->SetTikhonov(args_info.tikhonov_arg);

  conjugategradient->SetGeometry( geometryReader->GetOutputObject() );
  conjugategradient->SetNumberOfIterations( args_info.niterations_arg );
  conjugategradient->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

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

  if(args_info.costs_given)
    {
    costs=conjugategradient->GetResidualCosts();
    std::cout << "Residual costs at each iteration :" << std::endl;
    copy(costs.begin(),costs.end(),costs_it);
    }

  // Write
  typedef itk::ImageFileWriter< OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( conjugategradient->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
