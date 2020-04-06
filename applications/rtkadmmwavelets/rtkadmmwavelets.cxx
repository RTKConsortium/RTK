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

#include "rtkadmmwavelets_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include "rtkProjectionsReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkConstantImageSource.h"
#include "rtkDisplacedDetectorImageFilter.h"
#include "rtkADMMWaveletsConeBeamReconstructionFilter.h"

int
main(int argc, char * argv[])
{
  GGO(rtkadmmwavelets, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

  //////////////////////////////////////////////////////////////////////////////////////////
  // Read all the inputs
  //////////////////////////////////////////////////////////////////////////////////////////

  // Projections reader
  using projectionsReaderType = rtk::ProjectionsReader<OutputImageType>;
  projectionsReaderType::Pointer projectionsReader = projectionsReaderType::New();
  rtk::SetProjectionsReaderFromGgo<projectionsReaderType, args_info_rtkadmmwavelets>(projectionsReader, args_info);

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometryReader->GenerateOutputInformation());

  // Displaced detector weighting
  using DDFType = rtk::DisplacedDetectorImageFilter<OutputImageType>;
  DDFType::Pointer ddf = DDFType::New();
  ddf->SetInput(projectionsReader->GetOutput());
  ddf->SetGeometry(geometryReader->GetOutputObject());

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
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkadmmwavelets>(constantImageSource,
                                                                                           args_info);
    inputFilter = constantImageSource;
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  // Setup the ADMM filter and run it
  //////////////////////////////////////////////////////////////////////////////////////////

  // Set the reconstruction filter
  using ADMM_Wavelets_FilterType = rtk::ADMMWaveletsConeBeamReconstructionFilter<OutputImageType>;
  ADMM_Wavelets_FilterType::Pointer admmFilter = ADMM_Wavelets_FilterType::New();

  // Set the forward and back projection filters to be used inside admmFilter
  SetForwardProjectionFromGgo(args_info, admmFilter.GetPointer());
  SetBackProjectionFromGgo(args_info, admmFilter.GetPointer());

  // Set the geometry and interpolation weights
  admmFilter->SetGeometry(geometryReader->GetOutputObject());

  // Set all numerical parameters
  admmFilter->SetCG_iterations(args_info.CGiter_arg);
  admmFilter->SetAL_iterations(args_info.niterations_arg);
  admmFilter->SetAlpha(args_info.alpha_arg);
  admmFilter->SetBeta(args_info.beta_arg);
  admmFilter->SetNumberOfLevels(args_info.levels_arg);
  admmFilter->SetOrder(args_info.order_arg);

  // Set the inputs of the ADMM filter
  admmFilter->SetInput(0, inputFilter->GetOutput());
  admmFilter->SetInput(1, projectionsReader->GetOutput());

  admmFilter->SetDisableDisplacedDetectorFilter(args_info.nodisplaced_flag);

  REPORT_ITERATIONS(admmFilter, ADMM_Wavelets_FilterType, OutputImageType);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(admmFilter->Update())

  // Set writer and write the output
  using WriterType = itk::ImageFileWriter<OutputImageType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(admmFilter->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  return EXIT_SUCCESS;
}
