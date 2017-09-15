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

#include "rtkdualenergysimplexdecomposition_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "rtkSimplexSpectralProjectionsDecompositionImageFilter.h"

#include <itkImageRegionIterator.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkdualenergysimplexdecomposition, args_info);

  typedef float PixelValueType;
  const unsigned int Dimension = 3;

  typedef itk::VectorImage< PixelValueType, Dimension > DecomposedProjectionType;
  typedef itk::ImageFileReader<DecomposedProjectionType> DecomposedProjectionReaderType;
  typedef itk::ImageFileWriter<DecomposedProjectionType> DecomposedProjectionWriterType;

  typedef itk::VectorImage< PixelValueType, Dimension > DualEnergyProjectionsType;
  typedef itk::ImageFileReader< DualEnergyProjectionsType > DualEnergyProjectionReaderType;

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

  DualEnergyProjectionReaderType::Pointer dualEnergyProjectionReader = DualEnergyProjectionReaderType::New();
  dualEnergyProjectionReader->SetFileName( args_info.dual_arg );
  dualEnergyProjectionReader->Update();

  IncidentSpectrumReaderType::Pointer incidentSpectrumReaderHighEnergy = IncidentSpectrumReaderType::New();
  incidentSpectrumReaderHighEnergy->SetFileName( args_info.high_arg );
  incidentSpectrumReaderHighEnergy->Update();

  IncidentSpectrumReaderType::Pointer incidentSpectrumReaderLowEnergy = IncidentSpectrumReaderType::New();
  incidentSpectrumReaderLowEnergy->SetFileName( args_info.low_arg );
  incidentSpectrumReaderLowEnergy->Update();

  MaterialAttenuationsReaderType::Pointer materialAttenuationsReader = MaterialAttenuationsReaderType::New();
  materialAttenuationsReader->SetFileName( args_info.attenuations_arg );
  materialAttenuationsReader->Update();

  // If the detector response is given by the user, use it. Otherwise, assume it is included in the
  // incident spectrum, and fill the response with ones
  DetectorResponseReaderType::Pointer detectorResponseReader = DetectorResponseReaderType::New();
  DetectorResponseImageType::Pointer detectorImage;
  if(args_info.detector_given)
    {
    detectorResponseReader->SetFileName( args_info.detector_arg );
    detectorResponseReader->Update();
    detectorImage = detectorResponseReader->GetOutput();
    }
  else
    {
    rtk::ConstantImageSource<DetectorResponseImageType>::Pointer detectorSource = rtk::ConstantImageSource<DetectorResponseImageType>::New();
    DetectorResponseImageType::SizeType sourceSize;
    sourceSize[0] = 1;
    sourceSize[1] = incidentSpectrumReaderHighEnergy->GetOutput()->GetVectorLength();
    detectorSource->SetSize(sourceSize);
    detectorSource->SetConstant(1.0);
    detectorSource->Update();
    detectorImage = detectorSource->GetOutput();
    }

  // Get parameters from the images
  const unsigned int MaximumEnergy = incidentSpectrumReaderHighEnergy->GetOutput()->GetVectorLength();

  IncidentSpectrumImageType::IndexType indexIncident;
  indexIncident.Fill(0);
  if (incidentSpectrumReaderLowEnergy->GetOutput()->GetPixel(indexIncident).Size() != MaximumEnergy)
    itkGenericExceptionMacro(<< "Low energy incident spectrum image has vector size "
                             << incidentSpectrumReaderLowEnergy->GetOutput()->GetPixel(indexIncident).Size()
                             << ", should be "
                             << MaximumEnergy);

  if (detectorImage->GetLargestPossibleRegion().GetSize()[1] != MaximumEnergy)
    itkGenericExceptionMacro(<< "Detector response image has "
                             << detectorImage->GetLargestPossibleRegion().GetSize()[1]
                             << "energies, should have "
                             << MaximumEnergy);

  // Create and set the filter
  typedef rtk::SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionType,
                                                                    DualEnergyProjectionsType,
                                                                    IncidentSpectrumImageType,
                                                                    DetectorResponseImageType,
                                                                    MaterialAttenuationsImageType> SimplexFilterType;
  SimplexFilterType::Pointer simplex = SimplexFilterType::New();
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
  DecomposedProjectionWriterType::Pointer writer = DecomposedProjectionWriterType::New();
  writer->SetInput(simplex->GetOutput(0));
  writer->SetFileName(args_info.output_arg);
  writer->Update();

  return EXIT_SUCCESS;
}
