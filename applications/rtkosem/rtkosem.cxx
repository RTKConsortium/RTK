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

#include "rtkosem_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkOSEMConeBeamReconstructionFilter.h"
#include "rtkPhaseGatingImageFilter.h"
#include "rtkIterationCommands.h"

#ifdef RTK_USE_CUDA
#  include "itkCudaImage.h"
#endif

#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkosem, args_info);

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
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkosem>(reader, args_info);

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
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkosem>(constantImageSource, args_info);
    constantImageSource->SetConstant(1.);
    inputFilter = constantImageSource;
  }

  // OSEM reconstruction filter
  auto osem = rtk::OSEMConeBeamReconstructionFilter<OutputImageType>::New();

  // Set the forward and back projection filters
  SetForwardProjectionFromGgo(args_info, osem.GetPointer());
  SetBackProjectionFromGgo(args_info, osem.GetPointer());
  osem->SetInput(inputFilter->GetOutput());
  osem->SetInput(1, reader->GetOutput());
  osem->SetGeometry(geometry);
  if (args_info.betaregularization_given)
    osem->SetBetaRegularization(args_info.betaregularization_arg);

  osem->SetNumberOfIterations(args_info.niterations_arg);
  osem->SetNumberOfProjectionsPerSubset(args_info.nprojpersubset_arg);
  osem->SetStoreNormalizationImages(!args_info.nostorenormalizationimages_flag);
  REPORT_ITERATIONS(osem, rtk::OSEMConeBeamReconstructionFilter<OutputImageType>, OutputImageType)

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(osem->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
