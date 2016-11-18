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
#include "rtkSimplexDualEnergyProjectionsDecompositionImageFilter.h"

#include <itkImageRegionIterator.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkdualenergysimplexdecomposition, args_info);

  typedef double PixelValueType;
  const unsigned int Dimension = 3;

  typedef itk::Image< PixelValueType, Dimension > DecomposedProjectionType;
  typedef itk::VectorImage< PixelValueType, Dimension > VectorDecomposedProjectionType;
  typedef itk::ImageFileReader<DecomposedProjectionType> DecomposedProjectionReaderType;
  typedef itk::ImageFileWriter<DecomposedProjectionType> DecomposedProjectionWriterType;

  typedef itk::Image< PixelValueType, Dimension > MeasuredDataType;
  typedef itk::VectorImage< PixelValueType, Dimension > VectorMeasuredDataType;
  typedef itk::ImageFileReader< MeasuredDataType > MeasuredDataReaderType;

  typedef itk::VectorImage< PixelValueType, Dimension-1 > IncidentSpectrumImageType;
  typedef itk::ImageFileReader<IncidentSpectrumImageType> IncidentSpectrumReaderType;

  typedef itk::Image< PixelValueType, Dimension-1 > DetectorResponseImageType;
  typedef itk::ImageFileReader<DetectorResponseImageType> DetectorResponseReaderType;

  typedef itk::Image< PixelValueType, Dimension-1 > MaterialAttenuationsImageType;
  typedef itk::ImageFileReader<MaterialAttenuationsImageType> MaterialAttenuationsReaderType;

  // Read all inputs
  DecomposedProjectionReaderType::Pointer materialOneProjectionReader = DecomposedProjectionReaderType::New();
  materialOneProjectionReader->SetFileName( args_info.in_material_one_arg );
  materialOneProjectionReader->Update();

  DecomposedProjectionReaderType::Pointer materialTwoProjectionReader = DecomposedProjectionReaderType::New();
  materialTwoProjectionReader->SetFileName( args_info.in_material_two_arg );
  materialTwoProjectionReader->Update();

  MeasuredDataReaderType::Pointer measuredHighEnergyReader = MeasuredDataReaderType::New();
  measuredHighEnergyReader->SetFileName( args_info.measured_high_arg );
  measuredHighEnergyReader->Update();

  MeasuredDataReaderType::Pointer measuredLowEnergyReader = MeasuredDataReaderType::New();
  measuredLowEnergyReader->SetFileName( args_info.measured_low_arg );
  measuredLowEnergyReader->Update();

  IncidentSpectrumReaderType::Pointer highEnergySpectrumReader = IncidentSpectrumReaderType::New();
  highEnergySpectrumReader->SetFileName( args_info.sp_high_arg );
  highEnergySpectrumReader->Update();

  IncidentSpectrumReaderType::Pointer lowEnergySpectrumReader = IncidentSpectrumReaderType::New();
  lowEnergySpectrumReader->SetFileName( args_info.sp_low_arg );
  lowEnergySpectrumReader->Update();

  DetectorResponseReaderType::Pointer detectorResponseReader = DetectorResponseReaderType::New();
  detectorResponseReader->SetFileName( args_info.detector_arg );
  detectorResponseReader->Update();

  MaterialAttenuationsReaderType::Pointer materialAttenuationsReader = MaterialAttenuationsReaderType::New();
  materialAttenuationsReader->SetFileName( args_info.attenuations_arg );
  materialAttenuationsReader->Update();

  // Get parameters from the images
  const unsigned int MaximumEnergy = highEnergySpectrumReader->GetOutput()->GetVectorLength();

  IncidentSpectrumImageType::IndexType indexIncident;
  indexIncident.Fill(0);
  if (lowEnergySpectrumReader->GetOutput()->GetPixel(indexIncident).Size() != MaximumEnergy)
    itkGenericExceptionMacro(<< "Low energy incident spectrum image has vector size "
                             << lowEnergySpectrumReader->GetOutput()->GetPixel(indexIncident).Size()
                             << ", should be "
                             << MaximumEnergy);

  if (detectorResponseReader->GetOutput()->GetLargestPossibleRegion().GetSize()[0] != MaximumEnergy)
    itkGenericExceptionMacro(<< "Detector response image has "
                             << detectorResponseReader->GetOutput()->GetLargestPossibleRegion().GetSize()[0]
                             << "energies, should have "
                             << MaximumEnergy);

  // Merge some of the itk::Image inputs into itk::VectorImage
  // First the initial solution for the material decomposition
  DecomposedProjectionType::RegionType largest = materialOneProjectionReader->GetOutput()->GetLargestPossibleRegion();

  VectorDecomposedProjectionType::Pointer vectorDecomposedProjections = VectorDecomposedProjectionType::New();
  vectorDecomposedProjections->SetVectorLength(2);
  vectorDecomposedProjections->SetRegions(largest);
  vectorDecomposedProjections->Allocate();

  itk::ImageRegionIterator<VectorDecomposedProjectionType> vdpIt(vectorDecomposedProjections, largest);
  itk::ImageRegionIterator<DecomposedProjectionType> matOneIt(materialOneProjectionReader->GetOutput(), largest);
  itk::ImageRegionIterator<DecomposedProjectionType> matTwoIt(materialTwoProjectionReader->GetOutput(), largest);

  itk::VariableLengthVector<PixelValueType> temp;
  temp.SetSize(2);
  for(; !vdpIt.IsAtEnd(); ++vdpIt, ++matOneIt, ++matTwoIt)
    {
    temp[0] = matOneIt.Get();
    temp[1] = matTwoIt.Get();
    vdpIt.Set(temp);
    }

  // Then the measured projection data
  VectorMeasuredDataType::Pointer vectorMeasuredData = VectorMeasuredDataType::New();
  vectorMeasuredData->SetVectorLength(2);
  vectorMeasuredData->SetRegions(largest);
  vectorMeasuredData->Allocate();

  // If necessary, also perform the conversion to energies
  vnl_vector<PixelValueType> vnlDetectorResponse;
  vnlDetectorResponse.set_size(MaximumEnergy);
  if(args_info.lineintegrals_flag)
    {
    // Read the detector response as a vnl vector
    itk::ImageRegionIterator<DetectorResponseImageType> detIt (detectorResponseReader->GetOutput(),
                                                               detectorResponseReader->GetOutput()->GetLargestPossibleRegion());
    for(unsigned int i=0; i<MaximumEnergy; i++, ++detIt)
      vnlDetectorResponse[i] = detIt.Get();
    }

  itk::ImageRegionIterator<VectorMeasuredDataType> vmdIt(vectorMeasuredData, largest);
  itk::ImageRegionIterator<MeasuredDataType> measHighIt(measuredHighEnergyReader->GetOutput(), largest);
  itk::ImageRegionIterator<MeasuredDataType> measLowIt(measuredLowEnergyReader->GetOutput(), largest);
  itk::ImageRegionIterator<IncidentSpectrumImageType> spHighIt(highEnergySpectrumReader->GetOutput(),
                                                               highEnergySpectrumReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<IncidentSpectrumImageType> spLowIt(lowEnergySpectrumReader->GetOutput(),
                                                              lowEnergySpectrumReader->GetOutput()->GetLargestPossibleRegion());
  for(; !vmdIt.IsAtEnd(); ++vmdIt, ++measHighIt, ++measLowIt)
    {
    temp[0] = measHighIt.Get();
    temp[1] = measLowIt.Get();

    if (args_info.lineintegrals_flag)
      {
      if(spHighIt.IsAtEnd())
        spHighIt.GoToBegin();
      if(spLowIt.IsAtEnd())
        spLowIt.GoToBegin();

      vnl_vector<PixelValueType> vnlSpHigh(spHighIt.Get().GetDataPointer(), spHighIt.Get().GetSize());
      vnl_vector<PixelValueType> vnlSpLow(spLowIt.Get().GetDataPointer(), spLowIt.Get().GetSize());

      PixelValueType I_zero_High = dot_product(vnlDetectorResponse, vnlSpHigh);
      PixelValueType I_zero_Low = dot_product(vnlDetectorResponse, vnlSpLow);

      temp[0] = exp(-temp[0]) * I_zero_High;
      temp[1] = exp(-temp[1]) * I_zero_Low;

      ++spHighIt;
      ++spLowIt;
      }

      vmdIt.Set(temp);
    }

  // Create and set the filter
  typedef rtk::SimplexDualEnergyProjectionsDecompositionImageFilter<VectorDecomposedProjectionType,
                                                                    VectorMeasuredDataType,
                                                                    IncidentSpectrumImageType,
                                                                    DetectorResponseImageType,
                                                                    MaterialAttenuationsImageType> SimplexFilterType;
  SimplexFilterType::Pointer simplex = SimplexFilterType::New();
  simplex->SetInputDecomposedProjections(vectorDecomposedProjections);
  simplex->SetInputMeasuredProjections(vectorMeasuredData);
  simplex->SetHighEnergyIncidentSpectrum(highEnergySpectrumReader->GetOutput());
  simplex->SetLowEnergyIncidentSpectrum(lowEnergySpectrumReader->GetOutput());
  simplex->SetDetectorResponse(detectorResponseReader->GetOutput());
  simplex->SetMaterialAttenuations(materialAttenuationsReader->GetOutput());
  simplex->SetNumberOfIterations(args_info.niterations_arg);
  simplex->SetOptimizeWithRestarts(args_info.restarts_flag);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(simplex->Update())

  // Split the output into two distinct files
  DecomposedProjectionType::Pointer materialOneProjection = DecomposedProjectionType::New();
  materialOneProjection->SetRegions(largest);
  materialOneProjection->CopyInformation(materialOneProjectionReader->GetOutput());
  materialOneProjection->Allocate();
  DecomposedProjectionType::Pointer materialTwoProjection = DecomposedProjectionType::New();
  materialTwoProjection->SetRegions(largest);
  materialTwoProjection->CopyInformation(materialTwoProjectionReader->GetOutput());
  materialTwoProjection->Allocate();

  itk::ImageRegionIterator<VectorDecomposedProjectionType> out_vdpIt(simplex->GetOutput(0), largest);
  itk::ImageRegionIterator<DecomposedProjectionType> out_matOneIt(materialOneProjection, largest);
  itk::ImageRegionIterator<DecomposedProjectionType> out_matTwoIt(materialTwoProjection, largest);

  for(; !out_vdpIt.IsAtEnd(); ++out_vdpIt, ++out_matOneIt, ++out_matTwoIt)
    {
    out_matOneIt.Set(out_vdpIt.Get()[0]);
    out_matTwoIt.Set(out_vdpIt.Get()[1]);
    }

  // Write outputs
  DecomposedProjectionWriterType::Pointer writer = DecomposedProjectionWriterType::New();
  writer->SetInput(materialOneProjection);
  writer->SetFileName(args_info.out_material_one_arg);
  writer->Update();
  writer->SetInput(materialTwoProjection);
  writer->SetFileName(args_info.out_material_two_arg);
  writer->Update();

  return EXIT_SUCCESS;
}
