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

#include "rtkregularizedconjugategradient_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkRegularizedConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

#ifdef RTK_USE_CUDA
#  include "itkCudaImage.h"
#  include "rtkCudaConstantVolumeSource.h"
#endif
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkregularizedconjugategradient, args_info);

  using OutputPixelType = float;

#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, 3>;
#else
  using OutputImageType = itk::Image<OutputPixelType, 3>;
#endif

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkregularizedconjugategradient>(reader, args_info);

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
    auto inputReader = itk::ImageFileReader<OutputImageType>::New();
    inputReader->SetFileName(args_info.input_arg);
    inputFilter = inputReader;
  }
  else
  {
    // Create new empty volume
    using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
    auto constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkregularizedconjugategradient>(
      constantImageSource, args_info);
    inputFilter = constantImageSource;
  }
  TRY_AND_EXIT_ON_ITK_EXCEPTION(inputFilter->Update())
  inputFilter->ReleaseDataFlagOn();

  // Read weights if given, otherwise default to weights all equal to one
  itk::ImageSource<OutputImageType>::Pointer weightsSource;
  if (args_info.weights_given)
  {
    auto weightsReader = itk::ImageFileReader<OutputImageType>::New();
    weightsReader->SetFileName(args_info.weights_arg);
    weightsSource = weightsReader;
  }
  else
  {
    auto constantWeightsSource = rtk::ConstantImageSource<OutputImageType>::New();

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
    TRY_AND_EXIT_ON_ITK_EXCEPTION(supportmaskSource = itk::ReadImage<OutputImageType>(args_info.mask_arg))
  }

  // Set the forward and back projection filters to be used
  using ConjugateGradientFilterType = rtk::RegularizedConjugateGradientConeBeamReconstructionFilter<OutputImageType>;
  auto regularizedConjugateGradient = ConjugateGradientFilterType::New();
  SetForwardProjectionFromGgo(args_info, regularizedConjugateGradient.GetPointer());
  SetBackProjectionFromGgo(args_info, regularizedConjugateGradient.GetPointer());
  regularizedConjugateGradient->SetInputVolume(inputFilter->GetOutput());
  regularizedConjugateGradient->SetInputProjectionStack(reader->GetOutput());
  regularizedConjugateGradient->SetInputWeights(weightsSource->GetOutput());
  regularizedConjugateGradient->SetGeometry(geometry);
  regularizedConjugateGradient->SetMainLoop_iterations(args_info.niter_arg);
  regularizedConjugateGradient->SetCudaConjugateGradient(!args_info.nocudacg_flag);
  regularizedConjugateGradient->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);
  if (args_info.mask_given)
  {
    regularizedConjugateGradient->SetSupportMask(supportmaskSource);
  }

  // Positivity
  if (args_info.nopositivity_flag)
    regularizedConjugateGradient->SetPerformPositivity(false);
  else
    regularizedConjugateGradient->SetPerformPositivity(true);

  if (args_info.gammalaplacian_given)
    regularizedConjugateGradient->SetGamma(args_info.gammalaplacian_arg);
  if (args_info.tikhonov_given)
    regularizedConjugateGradient->SetTikhonov(args_info.tikhonov_arg);

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

  REPORT_ITERATIONS(regularizedConjugateGradient, ConjugateGradientFilterType, OutputImageType)

  TRY_AND_EXIT_ON_ITK_EXCEPTION(regularizedConjugateGradient->Update())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(regularizedConjugateGradient->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
