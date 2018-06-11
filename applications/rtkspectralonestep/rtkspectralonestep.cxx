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

#include "rtkspectralonestep_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkMechlemOneStepSpectralReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkReorderProjectionsImageFilter.h"

#include <algorithm>    // std::shuffle
#include <vector>       // std::vector
#include <random>       // std::default_random_engine
#include <iostream>
#include <iterator>

#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkspectralonestep, args_info);

  typedef float dataType;
  const unsigned int Dimension = 3;
  const unsigned int nBins = 5;
  const unsigned int nEnergies = 150;
  const unsigned int nMaterials = 3;

  // Define types for the input images
#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< itk::Vector<dataType, nMaterials>, Dimension > MaterialVolumesType;
  typedef itk::CudaImage< itk::Vector<dataType, nBins>, Dimension > PhotonCountsType;
  typedef itk::CudaImage< itk::Vector<dataType, nEnergies>, Dimension-1 > IncidentSpectrumType;
  typedef itk::CudaImage< dataType, 2 > DetectorResponseType;
  typedef itk::CudaImage< dataType, 2 > MaterialAttenuationsType;
#else
  typedef itk::Image< itk::Vector<dataType, nMaterials>, Dimension > MaterialVolumesType;
  typedef itk::Image< itk::Vector<dataType, nBins>, Dimension > PhotonCountsType;
  typedef itk::Image< itk::Vector<dataType, nEnergies>, Dimension-1 > IncidentSpectrumType;
  typedef itk::Image< dataType, 2 > DetectorResponseType;
  typedef itk::Image< dataType, 2 > MaterialAttenuationsType;
#endif

  // Define types for the readers
  typedef itk::ImageFileReader<MaterialVolumesType> MaterialVolumesReaderType;
  typedef itk::ImageFileReader<PhotonCountsType> PhotonCountsReaderType;
  typedef itk::ImageFileReader<IncidentSpectrumType> IncidentSpectrumReaderType;
  typedef itk::ImageFileReader<DetectorResponseType> DetectorResponseReaderType;
  typedef itk::ImageFileReader<MaterialAttenuationsType> MaterialAttenuationsReaderType;

  // Instantiate and update the readers
  PhotonCountsReaderType::Pointer photonCountsReader = PhotonCountsReaderType::New();
  photonCountsReader->SetFileName(args_info.spectral_arg);
  photonCountsReader->Update();

  IncidentSpectrumReaderType::Pointer incidentSpectrumReader = IncidentSpectrumReaderType::New();
  incidentSpectrumReader->SetFileName(args_info.incident_arg);
  incidentSpectrumReader->Update();

  DetectorResponseReaderType::Pointer detectorResponseReader = DetectorResponseReaderType::New();
  detectorResponseReader->SetFileName(args_info.detector_arg);
  detectorResponseReader->Update();

  MaterialAttenuationsReaderType::Pointer materialAttenuationsReader = MaterialAttenuationsReaderType::New();
  materialAttenuationsReader->SetFileName(args_info.attenuations_arg);
  materialAttenuationsReader->Update();

  // Create input: either an existing volume read from a file or a blank image
  itk::ImageSource< MaterialVolumesType >::Pointer inputFilter;
  if(args_info.input_given)
    {
    MaterialVolumesReaderType::Pointer materialVolumesReader = MaterialVolumesReaderType::New();
    materialVolumesReader->SetFileName(args_info.input_arg);
    inputFilter = materialVolumesReader;
    }
  else
    {
    // Create new empty volume
    typedef rtk::ConstantImageSource< MaterialVolumesType > ConstantImageSourceType;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkspectralonestep>(constantImageSource, args_info);
    inputFilter = constantImageSource;
    }
  inputFilter->Update();

  // Read the material attenuations image as a matrix
  MaterialAttenuationsType::IndexType indexMat;
  itk::Matrix<dataType, nEnergies, nMaterials> materialAttenuationsMatrix;
  for (unsigned int energy=0; energy<nEnergies; energy++)
    {
    indexMat[1] = energy;
    for (unsigned int material=0; material<nMaterials; material++)
      {
      indexMat[0] = material;
      materialAttenuationsMatrix[energy][material] = materialAttenuationsReader->GetOutput()->GetPixel(indexMat);
      }
    }

  // Read the thresholds on command line and check their number
  itk::Vector<unsigned int, nBins+1> thresholds;
  if (args_info.thresholds_given == nBins)
    {
    for (unsigned int bin=0; bin<nBins; bin++)
      thresholds[bin] = args_info.thresholds_arg[bin];

    // Add the maximum pulse height at the end
    unsigned int MaximumPulseHeight = detectorResponseReader->GetOutput()->GetLargestPossibleRegion().GetSize()[1];
    thresholds[nBins] = MaximumPulseHeight;
    }
  else
    itkGenericExceptionMacro(<< "Number of thresholds "<< args_info.thresholds_given << " does not match the number of bins " << nBins);

  // Read the detector response image as a matrix, and bin it
  itk::Matrix<dataType, nBins, nEnergies> detectorResponseMatrix;
  DetectorResponseType::IndexType indexDet;
  for (unsigned int energy=0; energy<nEnergies; energy++)
    {
    indexDet[0] = energy;
    for (unsigned int bin=0; bin<nBins; bin++)
      {
      for (unsigned int pulseHeight=thresholds[bin]-1; pulseHeight<thresholds[bin+1]; pulseHeight++)
        {
        indexDet[1] = pulseHeight;
        // Linear interpolation on the pulse heights: half of the pulses that have "threshold"
        // height are considered below threshold, the other half are considered above threshold
        if ((pulseHeight == thresholds[bin]-1) || (pulseHeight == thresholds[bin+1] - 1))
          detectorResponseMatrix[bin][energy] += detectorResponseReader->GetOutput()->GetPixel(indexDet) / 2;
        else
          detectorResponseMatrix[bin][energy] += detectorResponseReader->GetOutput()->GetPixel(indexDet);
        }
      }
    }

  // Geometry
  if(args_info.verbose_flag)
    std::cout << "Reading geometry information from "
              << args_info.geometry_arg
              << "..."
              << std::endl;
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() )

  // Read the regularization parameters
  MaterialVolumesType::RegionType::SizeType regulRadius;
  if(args_info.regul_radius_given)
    for(unsigned int i=0; i<Dimension; i++)
      regulRadius[i] = args_info.regul_radius_arg[i];
  else
    regulRadius.Fill(0);

  MaterialVolumesType::PixelType regulWeights;
  if(args_info.regul_weights_given)
    for(unsigned int i=0; i<Dimension; i++)
      regulWeights[i] = args_info.regul_weights_arg[i];
  else
    regulWeights.Fill(0);

  // Set the forward and back projection filters to be used
  typedef rtk::MechlemOneStepSpectralReconstructionFilter<MaterialVolumesType,
                                                          PhotonCountsType,
                                                          IncidentSpectrumType> MechlemFilterType;
  MechlemFilterType::Pointer mechlemOneStep = MechlemFilterType::New();
  SetForwardProjectionFromGgo(args_info, mechlemOneStep.GetPointer());
  SetBackProjectionFromGgo(args_info, mechlemOneStep.GetPointer());
  mechlemOneStep->SetInputMaterialVolumes( inputFilter->GetOutput() );
  mechlemOneStep->SetInputSpectrum(incidentSpectrumReader->GetOutput());
  mechlemOneStep->SetBinnedDetectorResponse(detectorResponseMatrix);
  mechlemOneStep->SetMaterialAttenuations(materialAttenuationsMatrix);
  mechlemOneStep->SetNumberOfIterations( args_info.niterations_arg );
  mechlemOneStep->SetNumberOfSubsets( args_info.subsets_arg );
  mechlemOneStep->SetRegularizationRadius( regulRadius );
  mechlemOneStep->SetRegularizationWeights( regulWeights );

  // If subsets are used, reorder projections and geometry according to
  // a random permutation
  if (args_info.subsets_arg != 1)
    {
    typedef rtk::ReorderProjectionsImageFilter<PhotonCountsType> ReorderProjectionsFilterType;
    ReorderProjectionsFilterType::Pointer reorder = ReorderProjectionsFilterType::New();
    reorder->SetInput(photonCountsReader->GetOutput());
    reorder->SetInputGeometry(geometryReader->GetOutputObject());
    reorder->SetPermutation(rtk::ReorderProjectionsImageFilter<PhotonCountsType>::SHUFFLE);
    TRY_AND_EXIT_ON_ITK_EXCEPTION( reorder->Update() )
    mechlemOneStep->SetInputPhotonCounts(reorder->GetOutput());
    mechlemOneStep->SetGeometry(reorder->GetOutputGeometry());
    }
  else
    {
    mechlemOneStep->SetInputPhotonCounts(photonCountsReader->GetOutput());
    mechlemOneStep->SetGeometry(geometryReader->GetOutputObject());
    }

  TRY_AND_EXIT_ON_ITK_EXCEPTION( mechlemOneStep->Update() )

  // Write
  typedef itk::ImageFileWriter< MaterialVolumesType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( mechlemOneStep->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
