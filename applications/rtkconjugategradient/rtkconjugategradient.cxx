/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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
#include "rtkIterationCommands.h"

#include <iostream>
#include <fstream>
#include <iterator>

#ifdef RTK_USE_CUDA
#  include <itkCudaImage.h>
#endif
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkconjugategradient, args_info);

  using OutputPixelType = float;
  constexpr unsigned int        Dimension = 3;
  std::vector<double>           costs;
  std::ostream_iterator<double> costs_it(std::cout, "\n");

#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkconjugategradient>(reader, args_info);

  // Projection matrix tolerance
  if (!args_info.tolerance_arg)
  {
    rtk::ThreeDCircularProjectionGeometryXMLFileReader::SetGeometryTolerance(args_info.tolerance_arg);
  }

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

  // Create input: either an existing volume read from a file or a blank image
  itk::ImageSource<OutputImageType>::Pointer inputFilter;
  if (args_info.input_given)
  {
    // Read an existing image to initialize the volume
    using InputReaderType = itk::ImageFileReader<OutputImageType>;
    InputReaderType::Pointer inputReader = InputReaderType::New();
    inputReader->SetFileName(args_info.input_arg);
    inputFilter = inputReader;
  }
  else
  {
    // Create new empty volume
    using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkconjugategradient>(constantImageSource,
                                                                                                args_info);
    inputFilter = constantImageSource;
  }

  // Read weights if given, otherwise default to weights all equal to one
  itk::ImageSource<OutputImageType>::Pointer weightsSource;
  if (args_info.weights_given)
  {
    using WeightsReaderType = itk::ImageFileReader<OutputImageType>;
    WeightsReaderType::Pointer weightsReader = WeightsReaderType::New();
    weightsReader->SetFileName(args_info.weights_arg);
    weightsSource = weightsReader;
  }
  else
  {
    using ConstantWeightsSourceType = rtk::ConstantImageSource<OutputImageType>;
    ConstantWeightsSourceType::Pointer constantWeightsSource = ConstantWeightsSourceType::New();

    // Set the weights to be like the projections
    TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->UpdateOutputInformation())
    constantWeightsSource->SetInformationFromImage(reader->GetOutput());
    constantWeightsSource->SetConstant(1.0);
    weightsSource = constantWeightsSource;
  }

  // Read Support Mask if given
  OutputImageType::Pointer supportmaskSource;
  if (args_info.mask_given)
  {
    supportmaskSource = itk::ReadImage<OutputImageType>(args_info.mask_arg);
  }

  // Set the forward and back projection filters to be used
  using ConjugateGradientFilterType = rtk::ConjugateGradientConeBeamReconstructionFilter<OutputImageType>;
  ConjugateGradientFilterType::Pointer conjugategradient = ConjugateGradientFilterType::New();
  SetForwardProjectionFromGgo(args_info, conjugategradient.GetPointer());
  SetBackProjectionFromGgo(args_info, conjugategradient.GetPointer());
  conjugategradient->SetInput(inputFilter->GetOutput());
  conjugategradient->SetInput(1, reader->GetOutput());
  conjugategradient->SetInput(2, weightsSource->GetOutput());
  conjugategradient->SetCudaConjugateGradient(!args_info.nocudacg_flag);
  if (args_info.mask_given)
  {
    conjugategradient->SetSupportMask(supportmaskSource);
  }
  //  conjugategradient->SetIterationCosts(args_info.costs_flag);

  if (args_info.gamma_given)
    conjugategradient->SetGamma(args_info.gamma_arg);
  if (args_info.tikhonov_given)
    conjugategradient->SetTikhonov(args_info.tikhonov_arg);

  conjugategradient->SetGeometry(geometry);
  conjugategradient->SetNumberOfIterations(args_info.niterations_arg);
  conjugategradient->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  REPORT_ITERATIONS(conjugategradient, ConjugateGradientFilterType, OutputImageType)

  TRY_AND_EXIT_ON_ITK_EXCEPTION(conjugategradient->Update())

  if (args_info.costs_given)
  {
    costs = conjugategradient->GetResidualCosts();
    std::cout << "Residual costs at each iteration :" << std::endl;
    copy(costs.begin(), costs.end(), costs_it);
  }

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(conjugategradient->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
