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
#  include <itkCovariantVector.h>
#endif
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>

int
main(int argc, char * argv[])
{
  GGO(rtkconjugategradient, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkconjugategradient>(reader, args_info);

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
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkconjugategradient>(constantImageSource,
                                                                                                args_info);
    inputFilter = constantImageSource;
  }

  // Read weights if given
  OutputImageType::Pointer inputWeights;
  if (args_info.weights_given)
  {
    using WeightsReaderType = itk::ImageFileReader<OutputImageType>;
    auto weightsReader = WeightsReaderType::New();
    weightsReader->SetFileName(args_info.weights_arg);
    inputWeights = weightsReader->GetOutput();
    TRY_AND_EXIT_ON_ITK_EXCEPTION(inputWeights->Update())
  }

  // Read regularization weights if given
  OutputImageType::Pointer localRegWeights;
  if (args_info.regweights_given)
  {
    using WeightsReaderType = itk::ImageFileReader<OutputImageType>;
    auto localRegWeightsReader = WeightsReaderType::New();
    localRegWeightsReader->SetFileName(args_info.regweights_arg);
    localRegWeights = localRegWeightsReader->GetOutput();
    localRegWeights->Update();
  }

  // Read Support Mask if given
  OutputImageType::Pointer supportmaskSource;
  if (args_info.mask_given)
  {
    supportmaskSource = itk::ReadImage<OutputImageType>(args_info.mask_arg);
  }

  // Set the forward and back projection filters to be used
  auto conjugategradient = rtk::ConjugateGradientConeBeamReconstructionFilter<OutputImageType>::New();
  SetForwardProjectionFromGgo(args_info, conjugategradient.GetPointer());
  SetBackProjectionFromGgo(args_info, conjugategradient.GetPointer());


#ifdef RTK_USE_CUDA
  {
    const bool warpRequested = (args_info.fp_arg == fp_arg_CudaWarpRayCast) || (args_info.bp_arg == bp_arg_CudaWarp);
    if (warpRequested && !args_info.dvf_given)
    {
      itkGenericExceptionMacro(<< "Warp projectors selected but no --dvf provided. Please provide "
                                  "a DVF in physical units (mm).");
    }
    if (args_info.dvf_given)
    {
      using VectorType = itk::CovariantVector<OutputPixelType, 3>;
      using DVF3DCudaType = itk::CudaImage<VectorType, 3>;

      auto dvf3dReader = itk::ImageFileReader<DVF3DCudaType>::New();
      dvf3dReader->SetFileName(args_info.dvf_arg);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(dvf3dReader->Update())

      conjugategradient->SetDisplacementField(reinterpret_cast<const OutputImageType *>(dvf3dReader->GetOutput()));
    }
  }
#endif

  conjugategradient->SetInputVolume(inputFilter->GetOutput());
  conjugategradient->SetInputProjectionStack(reader->GetOutput());
  conjugategradient->SetInputWeights(inputWeights);
  conjugategradient->SetLocalRegularizationWeights(localRegWeights);
  conjugategradient->SetCudaConjugateGradient(!args_info.nocudacg_flag);
  if (args_info.mask_given)
  {
    conjugategradient->SetSupportMask(supportmaskSource);
  }

  if (args_info.gamma_given)
    conjugategradient->SetGamma(args_info.gamma_arg);
  if (args_info.tikhonov_given)
    conjugategradient->SetTikhonov(args_info.tikhonov_arg);

  conjugategradient->SetGeometry(geometry);
  conjugategradient->SetNumberOfIterations(args_info.niterations_arg);
  conjugategradient->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  REPORT_ITERATIONS(
    conjugategradient, rtk::ConjugateGradientConeBeamReconstructionFilter<OutputImageType>, OutputImageType)

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

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(conjugategradient->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
