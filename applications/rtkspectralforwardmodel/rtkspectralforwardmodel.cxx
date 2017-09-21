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

#include "rtkspectralforwardmodel_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "rtkSpectralForwardModelImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkspectralforwardmodel, args_info);

  typedef float PixelValueType;
  const unsigned int Dimension = 3;

  typedef itk::VectorImage< PixelValueType, Dimension > DecomposedProjectionType;
  typedef itk::ImageFileReader<DecomposedProjectionType> DecomposedProjectionReaderType;

  typedef itk::VectorImage< PixelValueType, Dimension > SpectralProjectionsType;
  typedef itk::ImageFileWriter< SpectralProjectionsType > SpectralProjectionWriterType;

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
  const unsigned int NumberOfSpectralBins = args_info.thresholds_given;
  const unsigned int MaximumEnergy = incidentSpectrumReader->GetOutput()->GetVectorLength();

  // Generate a set of zero-filled photon count projections
  SpectralProjectionsType::Pointer photonCounts = SpectralProjectionsType::New();
  photonCounts->CopyInformation(decomposedProjectionReader->GetOutput());
  photonCounts->SetVectorLength(NumberOfSpectralBins);
  photonCounts->Allocate();

  // Read the thresholds on command line
  itk::VariableLengthVector<unsigned int> thresholds;
  thresholds.SetSize(NumberOfSpectralBins+1);
  for (unsigned int i=0; i<NumberOfSpectralBins; i++)
    thresholds[i] = args_info.thresholds_arg[i];

  // Add the maximum pulse height at the end
  unsigned int MaximumPulseHeight = detectorResponseReader->GetOutput()->GetLargestPossibleRegion().GetSize()[1];
  thresholds[NumberOfSpectralBins] = MaximumPulseHeight;

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
  if (photonCounts->GetPixel(indexSpect).Size() != NumberOfSpectralBins)
    itkGenericExceptionMacro(<< "Spectral projections (i.e. photon count data) image has vector size "
                             << photonCounts->GetPixel(indexSpect).Size()
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
  typedef rtk::SpectralForwardModelImageFilter<DecomposedProjectionType, SpectralProjectionsType, IncidentSpectrumImageType> ForwardModelFilterType;
  ForwardModelFilterType::Pointer forward = ForwardModelFilterType::New();
  forward->SetInputDecomposedProjections(decomposedProjectionReader->GetOutput());
  forward->SetInputMeasuredProjections(photonCounts);
  forward->SetInputIncidentSpectrum(incidentSpectrumReader->GetOutput());
  forward->SetDetectorResponse(detectorResponseReader->GetOutput());
  forward->SetMaterialAttenuations(materialAttenuationsReader->GetOutput());
  forward->SetThresholds(thresholds);
  forward->SetIsSpectralCT(true);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(forward->Update())

  // Write output
  SpectralProjectionWriterType::Pointer writer = SpectralProjectionWriterType::New();
  writer->SetInput(forward->GetOutput());
  writer->SetFileName(args_info.output_arg);
  writer->Update();

  return EXIT_SUCCESS;
}
