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

#include "rtkspectralsimplexdecomposition_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "rtkSimplexSpectralProjectionsDecompositionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkspectralsimplexdecomposition, args_info);

  using PixelValueType = float;
  constexpr unsigned int Dimension = 3;

  using DecomposedProjectionType = itk::VectorImage<PixelValueType, Dimension>;

  using SpectralProjectionsType = itk::VectorImage<PixelValueType, Dimension>;

  using IncidentSpectrumImageType = itk::Image<PixelValueType, Dimension>;

  using DetectorResponseImageType = itk::Image<PixelValueType, Dimension - 1>;

  using MaterialAttenuationsImageType = itk::Image<PixelValueType, Dimension - 1>;

  // Read all inputs
  DecomposedProjectionType::Pointer decomposedProjection;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(decomposedProjection = itk::ReadImage<DecomposedProjectionType>(args_info.input_arg))

  SpectralProjectionsType::Pointer spectralProjection;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(spectralProjection = itk::ReadImage<SpectralProjectionsType>(args_info.spectral_arg))

  IncidentSpectrumImageType::Pointer incidentSpectrum;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(incidentSpectrum = itk::ReadImage<IncidentSpectrumImageType>(args_info.incident_arg))

  DetectorResponseImageType::Pointer detectorResponse;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(detectorResponse = itk::ReadImage<DetectorResponseImageType>(args_info.detector_arg))

  MaterialAttenuationsImageType::Pointer materialAttenuations;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(materialAttenuations =
                                  itk::ReadImage<MaterialAttenuationsImageType>(args_info.attenuations_arg))

  // Get parameters from the images
  const unsigned int NumberOfMaterials = materialAttenuations->GetLargestPossibleRegion().GetSize()[0];
  const unsigned int NumberOfSpectralBins = spectralProjection->GetVectorLength();
  const unsigned int MaximumEnergy = incidentSpectrum->GetLargestPossibleRegion().GetSize()[0];

  // Read the thresholds on command line and check their number
  itk::VariableLengthVector<unsigned int> thresholds;
  thresholds.SetSize(NumberOfSpectralBins + 1);
  if (args_info.thresholds_given == NumberOfSpectralBins)
  {
    for (unsigned int i = 0; i < NumberOfSpectralBins; i++)
      thresholds[i] = args_info.thresholds_arg[i];

    // Add the maximum pulse height at the end
    unsigned int MaximumPulseHeight = detectorResponse->GetLargestPossibleRegion().GetSize()[1];
    thresholds[NumberOfSpectralBins] = MaximumPulseHeight;
  }
  else
    itkGenericExceptionMacro(<< "Number of thresholds " << args_info.thresholds_given
                             << " does not match the number of bins " << NumberOfSpectralBins);

  // Check that the inputs have the expected size
  DecomposedProjectionType::IndexType indexDecomp;
  indexDecomp.Fill(0);
  if (decomposedProjection->GetPixel(indexDecomp).Size() != NumberOfMaterials)
    itkGenericExceptionMacro(<< "Decomposed projections (i.e. initialization data) image has vector size "
                             << decomposedProjection->GetPixel(indexDecomp).Size() << ", should be "
                             << NumberOfMaterials);

  SpectralProjectionsType::IndexType indexSpect;
  indexSpect.Fill(0);
  if (spectralProjection->GetPixel(indexSpect).Size() != NumberOfSpectralBins)
    itkGenericExceptionMacro(<< "Spectral projections (i.e. photon count data) image has vector size "
                             << spectralProjection->GetPixel(indexSpect).Size() << ", should be "
                             << NumberOfSpectralBins);

  if (detectorResponse->GetLargestPossibleRegion().GetSize()[0] != MaximumEnergy)
    itkGenericExceptionMacro(<< "Detector response image has "
                             << detectorResponse->GetLargestPossibleRegion().GetSize()[0] << "energies, should have "
                             << MaximumEnergy);

  // Create and set the filter
  auto simplex = rtk::SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionType,
                                                                         SpectralProjectionsType,
                                                                         IncidentSpectrumImageType>::New();
  simplex->SetInputDecomposedProjections(decomposedProjection);
  simplex->SetGuessInitialization(args_info.guess_flag);
  simplex->SetInputMeasuredProjections(spectralProjection);
  simplex->SetInputIncidentSpectrum(incidentSpectrum);
  simplex->SetDetectorResponse(detectorResponse);
  simplex->SetMaterialAttenuations(materialAttenuations);
  simplex->SetThresholds(thresholds);
  simplex->SetNumberOfIterations(args_info.niterations_arg);
  simplex->SetOptimizeWithRestarts(args_info.restarts_flag);
  simplex->SetLogTransformEachBin(args_info.log_flag);
  simplex->SetIsSpectralCT(true);

  // Note: The simplex filter is set to perform several searches for each pixel,
  // with different initializations, and keep the best one (SetOptimizeWithRestart(true)).
  // While it may yield better results, these initializations are partially random,
  // which makes the output non-reproducible.
  // The default behavior, used for example in the tests, is not to use this feature
  // (SetOptimizeWithRestart(false)), which makes the output reproducible.

  if (args_info.weightsmap_given)
    simplex->SetOutputInverseCramerRaoLowerBound(true);

  if (args_info.fischer_given)
    simplex->SetOutputFischerMatrix(true);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(simplex->Update())

  // Write output
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(simplex->GetOutput(0), args_info.output_arg))

  // If requested, write the weightsmap
  if (args_info.weightsmap_given)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(simplex->GetOutput(1), args_info.weightsmap_arg))
  }

  // If requested, write the fisher information matrix
  if (args_info.fischer_given)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(simplex->GetOutput(2), args_info.fischer_arg))
  }

  return EXIT_SUCCESS;
}
