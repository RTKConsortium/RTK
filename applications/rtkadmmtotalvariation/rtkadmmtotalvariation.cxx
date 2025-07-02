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

#include "rtkadmmtotalvariation_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include "rtkProjectionsReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkConstantImageSource.h"
#include "rtkADMMTotalVariationConeBeamReconstructionFilter.h"
#include "rtkPhaseGatingImageFilter.h"

int
main(int argc, char * argv[])
{
  GGO(rtkadmmtotalvariation, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
  using GradientOutputImageType = itk::CudaImage<itk::CovariantVector<OutputPixelType, Dimension>, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  using GradientOutputImageType = itk::Image<itk::CovariantVector<OutputPixelType, Dimension>, Dimension>;
#endif

  //////////////////////////////////////////////////////////////////////////////////////////
  // Read all the inputs
  //////////////////////////////////////////////////////////////////////////////////////////

  // Projections reader
  using projectionsReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto projectionsReader = projectionsReaderType::New();
  rtk::SetProjectionsReaderFromGgo<projectionsReaderType, args_info_rtkadmmtotalvariation>(projectionsReader,
                                                                                           args_info);

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

  // Phase gating weights reader
  auto phaseGating = rtk::PhaseGatingImageFilter<OutputImageType>::New();
  if (args_info.phases_given)
  {
    phaseGating->SetPhasesFileName(args_info.phases_arg);
    phaseGating->SetGatingWindowWidth(args_info.windowwidth_arg);
    phaseGating->SetGatingWindowCenter(args_info.windowcenter_arg);
    phaseGating->SetGatingWindowShape(args_info.windowshape_arg);
    phaseGating->SetInputProjectionStack(projectionsReader->GetOutput());
    phaseGating->SetInputGeometry(geometry);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(phaseGating->Update())
  }

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
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkadmmtotalvariation>(constantImageSource,
                                                                                                 args_info);
    inputFilter = constantImageSource;
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  // Setup the ADMM filter and run it
  //////////////////////////////////////////////////////////////////////////////////////////

  // Set the reconstruction filter
  using ADMM_TV_FilterType =
    rtk::ADMMTotalVariationConeBeamReconstructionFilter<OutputImageType, GradientOutputImageType>;
  auto admmFilter = ADMM_TV_FilterType::New();

  // Set the forward and back projection filters to be used inside admmFilter
  SetForwardProjectionFromGgo(args_info, admmFilter.GetPointer());
  SetBackProjectionFromGgo(args_info, admmFilter.GetPointer());

  // Set all four numerical parameters
  admmFilter->SetCG_iterations(args_info.CGiter_arg);
  admmFilter->SetAL_iterations(args_info.niterations_arg);
  admmFilter->SetAlpha(args_info.alpha_arg);
  admmFilter->SetBeta(args_info.beta_arg);

  // Set the inputs of the ADMM filter
  admmFilter->SetInput(0, inputFilter->GetOutput());
  if (args_info.phases_given)
  {
    admmFilter->SetInput(1, phaseGating->GetOutput());
    admmFilter->SetGeometry(phaseGating->GetOutputGeometry());
    admmFilter->SetGatingWeights(phaseGating->GetGatingWeightsOnSelectedProjections());
  }
  else
  {
    admmFilter->SetInput(1, projectionsReader->GetOutput());
    admmFilter->SetGeometry(geometry);
  }
  admmFilter->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  REPORT_ITERATIONS(admmFilter, ADMM_TV_FilterType, OutputImageType)

  TRY_AND_EXIT_ON_ITK_EXCEPTION(admmFilter->Update())

  // Write the output
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(admmFilter->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
