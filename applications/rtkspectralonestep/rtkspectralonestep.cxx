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

#include "rtkspectralonestep_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkMechlemOneStepSpectralReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkReorderProjectionsImageFilter.h"
#include "rtkSpectralForwardModelImageFilter.h"

#include <algorithm> // std::shuffle
#include <vector>    // std::vector
#include <random>    // std::default_random_engine
#include <iostream>
#include <iterator>

#include <itkImageFileWriter.h>

itk::ImageIOBase::Pointer
GetFileHeader(const std::string & filename)
{
  itk::ImageIOBase::Pointer reader =
    itk::ImageIOFactory::CreateImageIO(filename.c_str(), itk::ImageIOFactory::IOFileModeEnum::ReadMode);
  if (!reader)
  {
    itkGenericExceptionMacro(<< "Could not read " << filename);
  }
  reader->SetFileName(filename);
  reader->ReadImageInformation();
  return reader;
}

namespace rtk
{
template <unsigned int VNumberOfBins, unsigned int VNumberOfMaterials>
void
rtkspectralonestep(const args_info_rtkspectralonestep & args_info)
{
  using dataType = float;
  constexpr unsigned int Dimension = 3;

  // Define types for the input images
#ifdef RTK_USE_CUDA
  using MaterialVolumesType = typename itk::CudaImage<itk::Vector<dataType, VNumberOfMaterials>, Dimension>;
  using MeasuredProjectionsType = typename itk::CudaImage<itk::Vector<dataType, VNumberOfBins>, Dimension>;
  using IncidentSpectrumType = itk::CudaImage<dataType, Dimension>;
  using DetectorResponseType = itk::CudaImage<dataType, 2>;
  using MaterialAttenuationsType = itk::CudaImage<dataType, 2>;
#else
  using MaterialVolumesType = typename itk::Image<itk::Vector<dataType, VNumberOfMaterials>, Dimension>;
  using MeasuredProjectionsType = typename itk::Image<itk::Vector<dataType, VNumberOfBins>, Dimension>;
  using IncidentSpectrumType = itk::Image<dataType, Dimension>;
  using DetectorResponseType = itk::Image<dataType, 2>;
  using MaterialAttenuationsType = itk::Image<dataType, 2>;
#endif

  // Instantiate and update the readers
  typename MeasuredProjectionsType::Pointer mea;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(mea = itk::ReadImage<MeasuredProjectionsType>(args_info.spectral_arg))

  IncidentSpectrumType::Pointer incidentSpectrum;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(incidentSpectrum = itk::ReadImage<IncidentSpectrumType>(args_info.incident_arg))

  DetectorResponseType::Pointer detectorResponse;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(detectorResponse = itk::ReadImage<DetectorResponseType>(args_info.detector_arg))

  MaterialAttenuationsType::Pointer materialAttenuations;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(materialAttenuations =
                                  itk::ReadImage<MaterialAttenuationsType>(args_info.attenuations_arg))

  // Read Support Mask if given
  IncidentSpectrumType::Pointer supportmask;
  if (args_info.mask_given)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(supportmask = itk::ReadImage<IncidentSpectrumType>(args_info.mask_arg))
  }

  // Read spatial regularization weights if given
  IncidentSpectrumType::Pointer spatialRegulWeighs;
  if (args_info.regul_spatial_weights_given)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(spatialRegulWeighs =
                                    itk::ReadImage<IncidentSpectrumType>(args_info.regul_spatial_weights_arg))
  }

  // Read projections weights if given
  IncidentSpectrumType::Pointer projectionWeights;
  if (args_info.projection_weights_given)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(projectionWeights =
                                    itk::ReadImage<IncidentSpectrumType>(args_info.projection_weights_arg))
  }

  // Create input: either an existing volume read from a file or a blank image
  typename MaterialVolumesType::Pointer input;
  if (args_info.input_given)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(input = itk::ReadImage<MaterialVolumesType>(args_info.input_arg))
  }
  else
  {
    // Create new empty volume
    using ConstantImageSourceType = typename rtk::ConstantImageSource<MaterialVolumesType>;
    auto constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkspectralonestep>(constantImageSource,
                                                                                              args_info);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(constantImageSource->Update())
    input = constantImageSource->GetOutput();
  }

  // Read the material attenuations image as a matrix
  MaterialAttenuationsType::IndexType indexMat;
  unsigned int                        nEnergies = materialAttenuations->GetLargestPossibleRegion().GetSize()[1];
  vnl_matrix<dataType>                materialAttenuationsMatrix(nEnergies, VNumberOfMaterials);
  for (unsigned int energy = 0; energy < nEnergies; energy++)
  {
    indexMat[1] = energy;
    for (unsigned int material = 0; material < VNumberOfMaterials; material++)
    {
      indexMat[0] = material;
      materialAttenuationsMatrix[energy][material] = materialAttenuations->GetPixel(indexMat);
    }
  }

  // Read the thresholds on command line and check their number
  itk::VariableLengthVector<double> thresholds;
  thresholds.SetSize(VNumberOfBins + 1);
  if (args_info.thresholds_given == VNumberOfBins)
  {
    for (unsigned int bin = 0; bin < VNumberOfBins; bin++)
      thresholds[bin] = args_info.thresholds_arg[bin];

    // Add the maximum pulse height at the end
    double MaximumPulseHeight = detectorResponse->GetLargestPossibleRegion().GetSize()[1];
    thresholds[VNumberOfBins] = MaximumPulseHeight;
  }
  else
  {
    itkGenericExceptionMacro(<< "Number of thresholds " << args_info.thresholds_given
                             << " does not match the number of bins " << VNumberOfBins);
  }

  // Read the detector response image as a matrix, and bin it
  vnl_matrix<dataType> drm =
    rtk::SpectralBinDetectorResponse<dataType>(detectorResponse.GetPointer(), thresholds, nEnergies);

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

  // Read the regularization parameters
  typename MaterialVolumesType::RegionType::SizeType regulRadius;
  if (args_info.regul_radius_given)
    for (unsigned int i = 0; i < Dimension; i++)
      regulRadius[i] = args_info.regul_radius_arg[std::min(i, args_info.regul_radius_given - 1)];
  else
    regulRadius.Fill(0);

  typename MaterialVolumesType::PixelType regulWeights;
  if (args_info.regul_weights_given)
    for (unsigned int i = 0; i < VNumberOfMaterials; i++)
      regulWeights[i] = args_info.regul_weights_arg[std::min(i, args_info.regul_weights_given - 1)];
  else
    regulWeights.Fill(0);

  // Set the forward and back projection filters to be used
  using MechlemFilterType = typename rtk::
    MechlemOneStepSpectralReconstructionFilter<MaterialVolumesType, MeasuredProjectionsType, IncidentSpectrumType>;
  auto mechlemOneStep = MechlemFilterType::New();
  SetForwardProjectionFromGgo(args_info, mechlemOneStep.GetPointer());
  SetBackProjectionFromGgo(args_info, mechlemOneStep.GetPointer());
  mechlemOneStep->SetInputMaterialVolumes(input);
  mechlemOneStep->SetInputIncidentSpectrum(incidentSpectrum);
  mechlemOneStep->SetBinnedDetectorResponse(drm);
  mechlemOneStep->SetMaterialAttenuations(materialAttenuationsMatrix);
  mechlemOneStep->SetNumberOfIterations(args_info.niterations_arg);
  mechlemOneStep->SetNumberOfSubsets(args_info.subsets_arg);
  mechlemOneStep->SetRegularizationRadius(regulRadius);
  mechlemOneStep->SetRegularizationWeights(regulWeights);
  if (args_info.reset_nesterov_given)
    mechlemOneStep->SetResetNesterovEvery(args_info.reset_nesterov_arg);
  if (args_info.mask_given)
    mechlemOneStep->SetSupportMask(supportmask);
  if (args_info.regul_spatial_weights_given)
    mechlemOneStep->SetSpatialRegularizationWeights(spatialRegulWeighs);
  mechlemOneStep->SetInputMeasuredProjections(mea);
  mechlemOneStep->SetGeometry(geometry);
  if (args_info.projection_weights_given)
    mechlemOneStep->SetProjectionWeights(projectionWeights);

  REPORT_ITERATIONS(mechlemOneStep, MechlemFilterType, MaterialVolumesType);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(mechlemOneStep->Update())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(mechlemOneStep->GetOutput(), args_info.output_arg))
}
} // namespace rtk

int
main(int argc, char * argv[])
{
  GGO(rtkspectralonestep, args_info);
  try
  {
    itk::ImageIOBase::Pointer headerInputMeasuredProjections = GetFileHeader(args_info.spectral_arg);
    unsigned int              nBins = headerInputMeasuredProjections->GetNumberOfComponents();
    itk::ImageIOBase::Pointer headerAttenuations = GetFileHeader(args_info.attenuations_arg);
    unsigned int              nMaterials = headerAttenuations->GetDimensions(0);

    if (nMaterials == 3 && nBins == 1)
      rtk::rtkspectralonestep<1, 3>(args_info);
    else if (nMaterials == 2 && nBins == 1)
      rtk::rtkspectralonestep<1, 2>(args_info);
    else if (nMaterials == 2 && nBins == 2)
      rtk::rtkspectralonestep<2, 2>(args_info);
    else if (nMaterials == 2 && nBins == 5)
      rtk::rtkspectralonestep<5, 2>(args_info);
    else if (nMaterials == 3 && nBins == 5)
      rtk::rtkspectralonestep<5, 3>(args_info);
    else
    {
      std::cerr << nMaterials << " materials and " << nBins << " bins is not handled" << std::endl;
      return EXIT_FAILURE;
    }
  }
  catch (itk::ExceptionObject & err)
  {
    std::cerr << "ExceptionObject caught in rtkspectraleonestep." << std::endl;
    std::cerr << err << std::endl;
    exit(EXIT_FAILURE);
  }
  return EXIT_SUCCESS;
}
