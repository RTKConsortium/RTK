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

#include "rtkvectorconjugategradient_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkConjugateGradientConeBeamReconstructionFilter.h"

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
  GGO(rtkvectorconjugategradient, args_info);

  constexpr unsigned int Dimension = 3;
  constexpr unsigned int nMaterials = 3;

  using DataType = float;
  using PixelType = itk::Vector<DataType, nMaterials>;
  using WeightsType = itk::Vector<DataType, nMaterials * nMaterials>;

  std::vector<double>           costs;
  std::ostream_iterator<double> costs_it(std::cout << std::setprecision(15), "\n");

#ifdef RTK_USE_CUDA
  using SingleComponentImageType = itk::CudaImage<DataType, Dimension>;
  using OutputImageType = itk::CudaImage<PixelType, Dimension>;
  using WeightsImageType = itk::CudaImage<WeightsType, Dimension>;
#else
  using SingleComponentImageType = itk::Image<DataType, Dimension>;
  using OutputImageType = itk::Image<PixelType, Dimension>;
  using WeightsImageType = itk::Image<WeightsType, Dimension>;
#endif

  // Projections reader
  OutputImageType::Pointer projections;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(projections = itk::ReadImage<OutputImageType>(args_info.projections_arg))

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
  OutputImageType::Pointer input;
  if (args_info.input_given)
  {
    // Read an existing image to initialize the volume
    TRY_AND_EXIT_ON_ITK_EXCEPTION(input = itk::ReadImage<OutputImageType>(args_info.input_arg))
  }
  else
  {
    // Create new empty volume
    using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkvectorconjugategradient>(
      constantImageSource, args_info);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(constantImageSource->Update())
    input = constantImageSource->GetOutput();
  }

  // Read weights if given, otherwise default to weights all equal to one
  WeightsImageType::Pointer weightsSource;
  if (args_info.weights_given)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(weightsSource = itk::ReadImage<WeightsImageType>(args_info.weights_arg))
  }
  else
  {
    using ConstantWeightsSourceType = rtk::ConstantImageSource<WeightsImageType>;
    ConstantWeightsSourceType::Pointer constantWeightsSource = ConstantWeightsSourceType::New();

    // Set the weights to the identity matrix
    constantWeightsSource->SetInformationFromImage(projections);
    WeightsType constantWeight = itk::NumericTraits<WeightsType>::ZeroValue(constantWeight);
    for (unsigned int i = 0; i < nMaterials; i++)
      constantWeight[i + i * nMaterials] = 1;

    constantWeightsSource->SetConstant(constantWeight);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(constantWeightsSource->Update())
    weightsSource = constantWeightsSource->GetOutput();
  }

  // Read Support Mask if given
  SingleComponentImageType::Pointer supportmask;
  if (args_info.mask_given)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(supportmask = itk::ReadImage<SingleComponentImageType>(args_info.mask_arg))
  }

  // Set the forward and back projection filters to be used
  using ConjugateGradientFilterType =
    rtk::ConjugateGradientConeBeamReconstructionFilter<OutputImageType, SingleComponentImageType, WeightsImageType>;
  ConjugateGradientFilterType::Pointer conjugategradient = ConjugateGradientFilterType::New();
  //  conjugategradient->SetForwardProjectionFilter(ConjugateGradientFilterType::JOSEPH);
  //  conjugategradient->SetBackProjectionFilter(ConjugateGradientFilterType::JOSEPH);
  SetForwardProjectionFromGgo(args_info, conjugategradient.GetPointer());
  SetBackProjectionFromGgo(args_info, conjugategradient.GetPointer());
  conjugategradient->SetInputVolume(input);
  conjugategradient->SetInputProjectionStack(projections);
  conjugategradient->SetInputWeights(weightsSource);
  conjugategradient->SetCudaConjugateGradient(!args_info.nocudacg_flag);
  if (args_info.mask_given)
  {
    conjugategradient->SetSupportMask(supportmask);
  }
  conjugategradient->SetIterationCosts(args_info.costs_flag);

  if (args_info.tikhonov_given)
    conjugategradient->SetTikhonov(args_info.tikhonov_arg);

  conjugategradient->SetGeometry(geometry);
  conjugategradient->SetNumberOfIterations(args_info.niterations_arg);
  conjugategradient->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  itk::TimeProbe readerProbe;
  if (args_info.time_flag)
  {
    std::cout << "Recording elapsed time... " << std::flush;
    readerProbe.Start();
  }

  TRY_AND_EXIT_ON_ITK_EXCEPTION(conjugategradient->Update())

  if (args_info.time_flag)
  {
    //    conjugategradient->PrintTiming(std::cout);
    readerProbe.Stop();
    std::cout << "It took...  " << readerProbe.GetMean() << ' ' << readerProbe.GetUnit() << std::endl;
  }

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
