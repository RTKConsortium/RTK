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

#include "rtkdualenergyforwardmodel_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "rtkSpectralForwardModelImageFilter.h"
#include "rtkConstantImageSource.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkdualenergyforwardmodel, args_info);

  typedef float PixelValueType;
  const unsigned int Dimension = 3;

  typedef itk::VectorImage< PixelValueType, Dimension > DecomposedProjectionType;
  typedef itk::ImageFileReader<DecomposedProjectionType> DecomposedProjectionReaderType;

  typedef itk::VectorImage< PixelValueType, Dimension > DualEnergyProjectionsType;
  typedef itk::ImageFileWriter< DualEnergyProjectionsType > DualEnergyProjectionWriterType;

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

  // Generate a set of zero-filled intensity projections
  DualEnergyProjectionsType::Pointer dualEnergyProjections = DualEnergyProjectionsType::New();
  dualEnergyProjections->CopyInformation(decomposedProjectionReader->GetOutput());
  dualEnergyProjections->SetVectorLength(2);
  dualEnergyProjections->Allocate();

  // Check that the inputs have the expected size
  DecomposedProjectionType::IndexType indexDecomp;
  indexDecomp.Fill(0);
  if (decomposedProjectionReader->GetOutput()->GetVectorLength() != 2)
    itkGenericExceptionMacro(<< "Decomposed projections (i.e. initialization data) image has vector length "
                             << decomposedProjectionReader->GetOutput()->GetVectorLength()
                             << ", should be 2");

  if (materialAttenuationsReader->GetOutput()->GetLargestPossibleRegion().GetSize()[1] != MaximumEnergy)
    itkGenericExceptionMacro(<< "Material attenuations image has "
                             << materialAttenuationsReader->GetOutput()->GetLargestPossibleRegion().GetSize()[1]
                             << "energies, should have "
                             << MaximumEnergy);

  // Create and set the filter
  typedef rtk::SpectralForwardModelImageFilter<DecomposedProjectionType, DualEnergyProjectionsType> ForwardModelFilterType;
  ForwardModelFilterType::Pointer forward = ForwardModelFilterType::New();
  forward->SetInputDecomposedProjections(decomposedProjectionReader->GetOutput());
  forward->SetInputMeasuredProjections(dualEnergyProjections);
  forward->SetInputIncidentSpectrum(incidentSpectrumReaderHighEnergy->GetOutput());
  forward->SetInputSecondIncidentSpectrum(incidentSpectrumReaderLowEnergy->GetOutput());
  forward->SetDetectorResponse(detectorImage);
  forward->SetMaterialAttenuations(materialAttenuationsReader->GetOutput());
  forward->SetIsSpectralCT(false);
  forward->SetComputeVariances(args_info.variances_given);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(forward->Update())

  // Write output
  DualEnergyProjectionWriterType::Pointer writer = DualEnergyProjectionWriterType::New();
  writer->SetInput(forward->GetOutput());
  writer->SetFileName(args_info.output_arg);
  writer->Update();

  // If requested, write the variances
  if (args_info.variances_given)
    {
    writer->SetInput(forward->GetOutput(1));
    writer->SetFileName(args_info.variances_arg);
    writer->Update();
    }

  return EXIT_SUCCESS;
}
