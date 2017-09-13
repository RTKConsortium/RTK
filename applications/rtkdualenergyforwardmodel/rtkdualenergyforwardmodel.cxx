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
#include "rtkDualEnergyForwardModelImageFilter.h"

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

  typedef itk::VectorImage< PixelValueType, Dimension-1 > SpectrumAndDetectorResponseImageType;
  typedef itk::ImageFileReader<SpectrumAndDetectorResponseImageType> SpectrumAndDetectorResponseReaderType;

  typedef itk::Image< PixelValueType, Dimension-1 > MaterialAttenuationsImageType;
  typedef itk::ImageFileReader<MaterialAttenuationsImageType> MaterialAttenuationsReaderType;

  // Read all inputs
  DecomposedProjectionReaderType::Pointer decomposedProjectionReader = DecomposedProjectionReaderType::New();
  decomposedProjectionReader->SetFileName( args_info.input_arg );
  decomposedProjectionReader->Update();

  SpectrumAndDetectorResponseReaderType::Pointer spectrumAndDetectorResponseReaderHighEnergy = SpectrumAndDetectorResponseReaderType::New();
  spectrumAndDetectorResponseReaderHighEnergy->SetFileName( args_info.high_arg );
  spectrumAndDetectorResponseReaderHighEnergy->Update();

  SpectrumAndDetectorResponseReaderType::Pointer spectrumAndDetectorResponseReaderLowEnergy = SpectrumAndDetectorResponseReaderType::New();
  spectrumAndDetectorResponseReaderLowEnergy->SetFileName( args_info.low_arg );
  spectrumAndDetectorResponseReaderLowEnergy->Update();

  MaterialAttenuationsReaderType::Pointer materialAttenuationsReader = MaterialAttenuationsReaderType::New();
  materialAttenuationsReader->SetFileName( args_info.attenuations_arg );
  materialAttenuationsReader->Update();

  // Get parameters from the images
  const unsigned int MaximumEnergy = spectrumAndDetectorResponseReaderHighEnergy->GetOutput()->GetVectorLength();

  // Generate a set of zero-filled photon count projections
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
  typedef rtk::DualEnergyForwardModelImageFilter<DecomposedProjectionType, DualEnergyProjectionsType> ForwardModelFilterType;
  ForwardModelFilterType::Pointer forward = ForwardModelFilterType::New();
  forward->SetInputDecomposedProjections(decomposedProjectionReader->GetOutput());
  forward->SetInputDualEnergyProjections(dualEnergyProjections);
  forward->SetInputSpectrumAndDetectorResponseHighEnergy(spectrumAndDetectorResponseReaderHighEnergy->GetOutput());
  forward->SetInputSpectrumAndDetectorResponseLowEnergy(spectrumAndDetectorResponseReaderLowEnergy->GetOutput());
  forward->SetMaterialAttenuations(materialAttenuationsReader->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION(forward->Update())

  // Write output
  DualEnergyProjectionWriterType::Pointer writer = DualEnergyProjectionWriterType::New();
  writer->SetInput(forward->GetOutput());
  writer->SetFileName(args_info.output_arg);
  writer->Update();

  return EXIT_SUCCESS;
}
