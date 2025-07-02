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

#include "rtkdualenergysimplexdecomposition_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "rtkSimplexSpectralProjectionsDecompositionImageFilter.h"

#include <itkImageRegionIterator.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkdualenergysimplexdecomposition, args_info);

  using PixelValueType = float;
  constexpr unsigned int Dimension = 3;

  using DecomposedProjectionType = itk::VectorImage<PixelValueType, Dimension>;

  using DualEnergyProjectionsType = itk::VectorImage<PixelValueType, Dimension>;

  using IncidentSpectrumImageType = itk::Image<PixelValueType, Dimension>;
  using IncidentSpectrumReaderType = itk::ImageFileReader<IncidentSpectrumImageType>;

  using DetectorResponseImageType = itk::Image<PixelValueType, Dimension - 1>;

  using MaterialAttenuationsImageType = itk::Image<PixelValueType, Dimension - 1>;

  // Read all inputs
  auto decomposedProjectionReader = itk::ImageFileReader<DecomposedProjectionType>::New();
  decomposedProjectionReader->SetFileName(args_info.input_arg);
  decomposedProjectionReader->Update();

  auto dualEnergyProjectionReader = itk::ImageFileReader<DualEnergyProjectionsType>::New();
  dualEnergyProjectionReader->SetFileName(args_info.dual_arg);
  dualEnergyProjectionReader->Update();

  auto incidentSpectrumReaderHighEnergy = IncidentSpectrumReaderType::New();
  incidentSpectrumReaderHighEnergy->SetFileName(args_info.high_arg);
  incidentSpectrumReaderHighEnergy->Update();

  auto incidentSpectrumReaderLowEnergy = IncidentSpectrumReaderType::New();
  incidentSpectrumReaderLowEnergy->SetFileName(args_info.low_arg);
  incidentSpectrumReaderLowEnergy->Update();

  auto materialAttenuationsReader = itk::ImageFileReader<MaterialAttenuationsImageType>::New();
  materialAttenuationsReader->SetFileName(args_info.attenuations_arg);
  materialAttenuationsReader->Update();

  // Get parameters from the images
  const unsigned int MaximumEnergy =
    incidentSpectrumReaderHighEnergy->GetOutput()->GetLargestPossibleRegion().GetSize(0);

  // If the detector response is given by the user, use it. Otherwise, assume it is included in the
  // incident spectrum, and fill the response with ones
  auto                               detectorResponseReader = itk::ImageFileReader<DetectorResponseImageType>::New();
  DetectorResponseImageType::Pointer detectorImage;
  if (args_info.detector_given)
  {
    detectorResponseReader->SetFileName(args_info.detector_arg);
    detectorResponseReader->Update();
    detectorImage = detectorResponseReader->GetOutput();
  }
  else
  {
    auto detectorSource = rtk::ConstantImageSource<DetectorResponseImageType>::New();
    detectorSource->SetSize(itk::MakeSize(1, MaximumEnergy));
    detectorSource->SetConstant(1.0);
    detectorSource->Update();
    detectorImage = detectorSource->GetOutput();
  }

  if (detectorImage->GetLargestPossibleRegion().GetSize()[1] != MaximumEnergy)
    itkGenericExceptionMacro(<< "Detector response image has " << detectorImage->GetLargestPossibleRegion().GetSize()[1]
                             << "energies, should have " << MaximumEnergy);

  // Create and set the filter
  auto simplex = rtk::SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionType,
                                                                         DualEnergyProjectionsType,
                                                                         IncidentSpectrumImageType,
                                                                         DetectorResponseImageType,
                                                                         MaterialAttenuationsImageType>::New();
  simplex->SetInputDecomposedProjections(decomposedProjectionReader->GetOutput());
  simplex->SetGuessInitialization(args_info.guess_flag);
  simplex->SetInputMeasuredProjections(dualEnergyProjectionReader->GetOutput());
  simplex->SetInputIncidentSpectrum(incidentSpectrumReaderHighEnergy->GetOutput());
  simplex->SetInputSecondIncidentSpectrum(incidentSpectrumReaderLowEnergy->GetOutput());
  simplex->SetDetectorResponse(detectorImage);
  simplex->SetMaterialAttenuations(materialAttenuationsReader->GetOutput());
  simplex->SetNumberOfIterations(args_info.niterations_arg);
  simplex->SetOptimizeWithRestarts(args_info.restarts_flag);
  simplex->SetIsSpectralCT(false);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(simplex->Update())

  // Write outputs
  auto writer = itk::ImageFileWriter<DecomposedProjectionType>::New();
  writer->SetInput(simplex->GetOutput(0));
  writer->SetFileName(args_info.output_arg);
  writer->Update();

  return EXIT_SUCCESS;
}
