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

#include "rtkspectralsimplexdecomposition_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "rtkSimplexSpectralProjectionsDecompositionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkspectralsimplexdecomposition, args_info);

  typedef float PixelValueType;
  const unsigned int Dimension = 3;

  typedef itk::VectorImage< PixelValueType, Dimension > DecomposedProjectionType;
  typedef itk::ImageFileReader<DecomposedProjectionType> DecomposedProjectionReaderType;
  typedef itk::ImageFileWriter<DecomposedProjectionType> DecomposedProjectionWriterType;

  typedef itk::VectorImage< PixelValueType, Dimension > SpectralProjectionsType;
  typedef itk::ImageFileReader< SpectralProjectionsType > SpectralProjectionReaderType;

  typedef itk::VectorImage< PixelValueType, Dimension-1 > IncidentSpectrumImageType;
  typedef itk::ImageFileReader<IncidentSpectrumImageType> IncidentSpectrumReaderType;

  typedef itk::Image< PixelValueType, Dimension-1 > DetectorResponseImageType;
  typedef itk::ImageFileReader<DetectorResponseImageType> DetectorResponseReaderType;

  typedef itk::Image< PixelValueType, Dimension-1 > MaterialAttenuationsImageType;
  typedef itk::ImageFileReader<MaterialAttenuationsImageType> MaterialAttenuationsReaderType;

  // Read all inputs
  DecomposedProjectionReaderType::Pointer decomposedProjectionReader = DecomposedProjectionReaderType::New();
  decomposedProjectionReader->SetFileName( args_info.input_arg );
  decomposedProjectionReader->Update();

  SpectralProjectionReaderType::Pointer spectralProjectionReader = SpectralProjectionReaderType::New();
  spectralProjectionReader->SetFileName( args_info.spectral_arg );
  spectralProjectionReader->Update();

  IncidentSpectrumReaderType::Pointer incidentSpectrumReader = IncidentSpectrumReaderType::New();
  incidentSpectrumReader->SetFileName( args_info.incident_arg );
  incidentSpectrumReader->Update();

  DetectorResponseReaderType::Pointer detectorResponseReader = DetectorResponseReaderType::New();
  detectorResponseReader->SetFileName( args_info.detector_arg );
  detectorResponseReader->Update();

  MaterialAttenuationsReaderType::Pointer materialAttenuationsReader = MaterialAttenuationsReaderType::New();
  materialAttenuationsReader->SetFileName( args_info.attenuations_arg );
  materialAttenuationsReader->Update();

  // Get parameters from the images
  const unsigned int NumberOfMaterials = materialAttenuationsReader->GetOutput()->GetLargestPossibleRegion().GetSize()[0];
  const unsigned int NumberOfSpectralBins = spectralProjectionReader->GetOutput()->GetVectorLength();
  const unsigned int MaximumEnergy = incidentSpectrumReader->GetOutput()->GetVectorLength();

  // Read the thresholds on command line and check their number
  itk::VariableLengthVector<unsigned int> thresholds;
  thresholds.SetSize(NumberOfSpectralBins+1);
  if (args_info.thresholds_given == NumberOfSpectralBins)
    {
    for (unsigned int i=0; i<NumberOfSpectralBins; i++)
      thresholds[i] = args_info.thresholds_arg[i];

    // Add the maximum pulse height at the end
    unsigned int MaximumPulseHeight = detectorResponseReader->GetOutput()->GetLargestPossibleRegion().GetSize()[1];
    thresholds[NumberOfSpectralBins] = MaximumPulseHeight;
    }
  else
    itkGenericExceptionMacro(<< "Number of thresholds "<< args_info.thresholds_given << " does not match the number of bins " << NumberOfSpectralBins);

  // Check that the inputs have the expected size
  DecomposedProjectionType::IndexType indexDecomp;
  indexDecomp.Fill(0);
  if (decomposedProjectionReader->GetOutput()->GetPixel(indexDecomp).Size() != NumberOfMaterials)
    itkGenericExceptionMacro(<< "Decomposed projections (i.e. initialization data) image has vector size "
                             << decomposedProjectionReader->GetOutput()->GetPixel(indexDecomp).Size()
                             << ", should be "
                             << NumberOfMaterials);

  SpectralProjectionsType::IndexType indexSpect;
  indexSpect.Fill(0);
  if (spectralProjectionReader->GetOutput()->GetPixel(indexSpect).Size() != NumberOfSpectralBins)
    itkGenericExceptionMacro(<< "Spectral projections (i.e. photon count data) image has vector size "
                             << spectralProjectionReader->GetOutput()->GetPixel(indexSpect).Size()
                             << ", should be "
                             << NumberOfSpectralBins);

  IncidentSpectrumImageType::IndexType indexIncident;
  indexIncident.Fill(0);
  if (incidentSpectrumReader->GetOutput()->GetPixel(indexIncident).Size() != MaximumEnergy)
    itkGenericExceptionMacro(<< "Incident spectrum image has vector size "
                             << incidentSpectrumReader->GetOutput()->GetPixel(indexIncident).Size()
                             << ", should be "
                             << MaximumEnergy);

  if (detectorResponseReader->GetOutput()->GetLargestPossibleRegion().GetSize()[0] != MaximumEnergy)
    itkGenericExceptionMacro(<< "Detector response image has "
                             << detectorResponseReader->GetOutput()->GetLargestPossibleRegion().GetSize()[0]
                             << "energies, should have "
                             << MaximumEnergy);

  // Create and set the filter
  typedef rtk::SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionType,
                                                                  SpectralProjectionsType,
                                                                  IncidentSpectrumImageType> SimplexFilterType;
  SimplexFilterType::Pointer simplex = SimplexFilterType::New();
  simplex->SetInputDecomposedProjections(decomposedProjectionReader->GetOutput());
  simplex->SetGuessInitialization(args_info.guess_flag);
  simplex->SetInputMeasuredProjections(spectralProjectionReader->GetOutput());
  simplex->SetInputIncidentSpectrum(incidentSpectrumReader->GetOutput());
  simplex->SetDetectorResponse(detectorResponseReader->GetOutput());
  simplex->SetMaterialAttenuations(materialAttenuationsReader->GetOutput());
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
  DecomposedProjectionWriterType::Pointer writer = DecomposedProjectionWriterType::New();
  writer->SetInput(simplex->GetOutput(0));
  writer->SetFileName(args_info.output_arg);
  writer->Update();

  // If requested, write the weightsmap
  if (args_info.weightsmap_given)
    {
    writer->SetInput(simplex->GetOutput(1));
    writer->SetFileName(args_info.weightsmap_arg);
    writer->Update();
    }

  // If requested, write the fisher information matrix
  if (args_info.fischer_given)
    {
    writer->SetInput(simplex->GetOutput(2));
    writer->SetFileName(args_info.fischer_arg);
    writer->Update();
    }

  return EXIT_SUCCESS;
}
