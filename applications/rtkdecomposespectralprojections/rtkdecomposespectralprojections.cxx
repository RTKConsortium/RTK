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

#include "rtkdecomposespectralprojections_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "rtkSimplexSpectralProjectionsDecompositionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkdecomposespectralprojections, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  const unsigned int NumberOfMaterials = 3;
  const unsigned int NumberOfSpectralBins = 6;
  const unsigned int MaximumEnergy = 150;

  typedef itk::Vector<float, NumberOfMaterials> MaterialsVectorType;
  typedef itk::Image< MaterialsVectorType, Dimension > DecomposedProjectionType;
  typedef itk::ImageFileReader<DecomposedProjectionType> DecomposedProjectionReaderType;
  typedef itk::ImageFileWriter<DecomposedProjectionType> DecomposedProjectionWriterType;

  typedef itk::Vector<float, NumberOfSpectralBins> SpectralVectorType;
  typedef itk::Image< SpectralVectorType, Dimension > SpectralProjectionsType;
  typedef itk::ImageFileReader<SpectralProjectionsType> SpectralProjectionReaderType;

  typedef itk::Vector<float, MaximumEnergy> IncidentSpectrumVectorType;
  typedef itk::Image< IncidentSpectrumVectorType, Dimension-1 > IncidentSpectrumImageType;
  typedef itk::ImageFileReader<IncidentSpectrumImageType> IncidentSpectrumReaderType;

  typedef itk::Image< float, Dimension-1 > DetectorResponseImageType;
  typedef itk::ImageFileReader<DetectorResponseImageType> DetectorResponseReaderType;

  typedef itk::Image< float, Dimension-1 > MaterialAttenuationsImageType;
  typedef itk::ImageFileReader<MaterialAttenuationsImageType> MaterialAttenuationsReaderType;

  typedef itk::Matrix<float, NumberOfSpectralBins, MaximumEnergy>            DetectorResponseType;
  typedef itk::Vector<itk::Vector<float, MaximumEnergy>, NumberOfMaterials>  MaterialAttenuationsType;

  // Read all inputs
  DecomposedProjectionReaderType::Pointer DecomposedProjectionReader = DecomposedProjectionReaderType::New();
  DecomposedProjectionReader->SetFileName( args_info.input_arg );
  DecomposedProjectionReader->Update();

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

  // Check that the inputs have the expected size
  DecomposedProjectionType::IndexType indexDecomp;
  indexDecomp.Fill(0);
  if (DecomposedProjectionReader->GetOutput()->GetPixel(indexDecomp).Size() != NumberOfMaterials)
    itkGenericExceptionMacro(<< "Decomposed projections (i.e. initialization data) image has vector size "
                             << DecomposedProjectionReader->GetOutput()->GetPixel(indexDecomp).Size()
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

  if (detectorResponseReader->GetOutput()->GetLargestPossibleRegion().GetSize()[1] != NumberOfSpectralBins)
    itkGenericExceptionMacro(<< "Detector response image has "
                             << detectorResponseReader->GetOutput()->GetLargestPossibleRegion().GetSize()[0]
                             << "energy bins, should have "
                             << NumberOfSpectralBins);

  // Format the inputs to pass them to the filter
  // First the detector response matrix
  DetectorResponseType detectorResponseMatrix;
  DetectorResponseImageType::IndexType indexDet;
  for (unsigned int energy=0; energy<MaximumEnergy; energy++)
    {
    indexDet[0] = energy;
    for (unsigned int bin=0; bin<NumberOfSpectralBins; bin++)
      {
      indexDet[1] = bin;
      detectorResponseMatrix[bin][energy] = detectorResponseReader->GetOutput()->GetPixel(indexDet);
      }
    }

  // Then the material attenuations vector of vectors
  MaterialAttenuationsType materialAttenuationsVector;
  MaterialAttenuationsImageType::IndexType indexMat;
  for (unsigned int energy=0; energy<MaximumEnergy; energy++)
    {
    indexMat[1] = energy;
    for (unsigned int material=0; material<NumberOfMaterials; material++)
      {
      indexMat[0] = material;
      materialAttenuationsVector[material][energy] = materialAttenuationsReader->GetOutput()->GetPixel(indexMat);
      }
    }

  // Create and set the filter
  typedef rtk::SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionType, SpectralProjectionsType, MaximumEnergy, IncidentSpectrumImageType> SimplexFilterType;
  SimplexFilterType::Pointer simplex = SimplexFilterType::New();
  simplex->SetInputDecomposedProjections(DecomposedProjectionReader->GetOutput());
  simplex->SetInputSpectralProjections(spectralProjectionReader->GetOutput());
  simplex->SetInputIncidentSpectrum(incidentSpectrumReader->GetOutput());
  simplex->SetDetectorResponse(detectorResponseMatrix);
  simplex->SetMaterialAttenuations(materialAttenuationsVector);
  simplex->SetNumberOfIterations(args_info.niterations_arg);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(simplex->Update())

  // Write output
  DecomposedProjectionWriterType::Pointer writer = DecomposedProjectionWriterType::New();
  writer->SetInput(simplex->GetOutput());
  writer->SetFileName(args_info.output_arg);
  writer->Update();

  return EXIT_SUCCESS;
}
