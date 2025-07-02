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

#include "rtkspectralforwardmodel_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "rtkSpectralForwardModelImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkspectralforwardmodel, args_info);

  using PixelValueType = float;
  constexpr unsigned int Dimension = 3;
  using DecomposedProjectionType = itk::VectorImage<PixelValueType, Dimension>;
  using MeasuredProjectionsType = itk::VectorImage<PixelValueType, Dimension>;
  using IncidentSpectrumImageType = itk::Image<PixelValueType, Dimension>;
  using DetectorResponseImageType = itk::Image<PixelValueType, Dimension - 1>;
  using MaterialAttenuationsImageType = itk::Image<PixelValueType, Dimension - 1>;

  // Read all inputs
  DecomposedProjectionType::Pointer decomposedProjection;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(decomposedProjection = itk::ReadImage<DecomposedProjectionType>(args_info.input_arg))

  IncidentSpectrumImageType::Pointer incidentSpectrum;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(incidentSpectrum = itk::ReadImage<IncidentSpectrumImageType>(args_info.incident_arg))

  DetectorResponseImageType::Pointer detectorResponse;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(detectorResponse = itk::ReadImage<DetectorResponseImageType>(args_info.detector_arg))

  MaterialAttenuationsImageType::Pointer materialAttenuations;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(materialAttenuations =
                                  itk::ReadImage<MaterialAttenuationsImageType>(args_info.attenuations_arg))

  // Get parameters from the images
  const unsigned int NumberOfMaterials = materialAttenuations->GetLargestPossibleRegion().GetSize()[0];
  const unsigned int NumberOfSpectralBins = args_info.thresholds_given;
  const unsigned int MaximumEnergy = incidentSpectrum->GetLargestPossibleRegion().GetSize()[0];

  // Generate a set of zero-filled photon count projections
  auto measuredProjections = MeasuredProjectionsType::New();
  measuredProjections->CopyInformation(decomposedProjection);
  measuredProjections->SetVectorLength(NumberOfSpectralBins);
  measuredProjections->Allocate();

  // Read the thresholds on command line
  itk::VariableLengthVector<unsigned int> thresholds;
  thresholds.SetSize(NumberOfSpectralBins + 1);
  for (unsigned int i = 0; i < NumberOfSpectralBins; i++)
    thresholds[i] = args_info.thresholds_arg[i];

  // Add the maximum pulse height at the end
  unsigned int MaximumPulseHeight = detectorResponse->GetLargestPossibleRegion().GetSize()[1];
  thresholds[NumberOfSpectralBins] = MaximumPulseHeight;

  // Check that the inputs have the expected size
  DecomposedProjectionType::IndexType indexDecomp;
  indexDecomp.Fill(0);
  if (decomposedProjection->GetPixel(indexDecomp).Size() != NumberOfMaterials)
    itkGenericExceptionMacro(<< "Decomposed projections (i.e. initialization data) image has vector size "
                             << decomposedProjection->GetPixel(indexDecomp).Size() << ", should be "
                             << NumberOfMaterials);

  MeasuredProjectionsType::IndexType indexSpect;
  indexSpect.Fill(0);
  if (measuredProjections->GetPixel(indexSpect).Size() != NumberOfSpectralBins)
    itkGenericExceptionMacro(<< "Spectral projections (i.e. photon count data) image has vector size "
                             << measuredProjections->GetPixel(indexSpect).Size() << ", should be "
                             << NumberOfSpectralBins);

  if (detectorResponse->GetLargestPossibleRegion().GetSize()[0] != MaximumEnergy)
    itkGenericExceptionMacro(<< "Detector response image has "
                             << detectorResponse->GetLargestPossibleRegion().GetSize()[0] << "energies, should have "
                             << MaximumEnergy);

  // Create and set the filter
  auto forward = rtk::SpectralForwardModelImageFilter<DecomposedProjectionType,
                                                      MeasuredProjectionsType,
                                                      IncidentSpectrumImageType>::New();
  forward->SetInputDecomposedProjections(decomposedProjection);
  forward->SetInputMeasuredProjections(measuredProjections);
  forward->SetInputIncidentSpectrum(incidentSpectrum);
  forward->SetDetectorResponse(detectorResponse);
  forward->SetMaterialAttenuations(materialAttenuations);
  forward->SetThresholds(thresholds);
  forward->SetIsSpectralCT(true);
  if (args_info.cramer_rao_given)
    forward->SetComputeCramerRaoLowerBound(true);
  if (args_info.variances_given)
    forward->SetComputeVariances(true);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(forward->Update())

  // Write output
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(forward->GetOutput(), args_info.output_arg))

  // If requested, write the Cramer-Rao lower bound
  if (args_info.cramer_rao_given)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(forward->GetOutputCramerRaoLowerBound(), args_info.cramer_rao_arg))
  }

  // If requested, write the variance
  if (args_info.variances_given)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(forward->GetOutputVariances(), args_info.variances_arg))
  }


  return EXIT_SUCCESS;
}
