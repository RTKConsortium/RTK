#include "rtkTest.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkMechlemOneStepSpectralReconstructionFilter.h"
#include "rtkSpectralForwardModelImageFilter.h"
#include "rtkReorderProjectionsImageFilter.h"

#ifdef USE_CUDA
#  include "itkCudaImage.h"
#endif

#include <itkImageFileReader.h>
#include <itkComposeImageFilter.h>
#include <itkImportImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>

/**
 * \file rtkspectralonesteptest.cxx
 *
 * \brief Functional test for Mechlem one-step spectral reconstruction
 *
 * This test generates the projections of a multi-material phantom and reconstructs
 * from them using the MechlemOneStep algorithm with different backprojectors (Voxel-Based,
 * Joseph).
 *
 * \author Cyril Mory
 */

int
main(int argc, char * argv[])
{
  if (argc < 4)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  incident_spectrum  detector_response  material_attenuations" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 3;
  constexpr unsigned int nBins = 5;
  constexpr unsigned int nMaterials = 3;
  constexpr unsigned int nEnergies = 150;
  using DataType = float;
  using MaterialPixelType = itk::Vector<DataType, nMaterials>;
  using PhotonCountsPixelType = itk::Vector<DataType, nBins>;

  using MaterialProjectionsType = itk::VectorImage<DataType, Dimension>;
  using vIncidentSpectrum = itk::VectorImage<DataType, Dimension - 1>;
#ifdef RTK_USE_CUDA
  using MaterialVolumeType = itk::CudaImage<MaterialPixelType, Dimension>;
  using PhotonCountsType = itk::CudaImage<PhotonCountsPixelType, Dimension>;
  using IncidentSpectrumImageType = itk::CudaImage<DataType, Dimension>;
  using DetectorResponseImageType = itk::CudaImage<DataType, Dimension - 1>;
  using MaterialAttenuationsImageType = itk::CudaImage<DataType, Dimension - 1>;
  using SingleComponentImageType = itk::CudaImage<DataType, Dimension>;
#else
  using MaterialVolumeType = itk::Image<MaterialPixelType, Dimension>;
  using PhotonCountsType = itk::Image<PhotonCountsPixelType, Dimension>;
  using IncidentSpectrumImageType = itk::Image<DataType, Dimension>;
  using DetectorResponseImageType = itk::Image<DataType, Dimension - 1>;
  using MaterialAttenuationsImageType = itk::Image<DataType, Dimension - 1>;
  using SingleComponentImageType = itk::Image<DataType, Dimension>;
#endif

  using IncidentSpectrumReaderType = itk::ImageFileReader<IncidentSpectrumImageType>;
  using vIncidentSpectrumReaderType = itk::ImageFileReader<vIncidentSpectrum>;
  using DetectorResponseReaderType = itk::ImageFileReader<DetectorResponseImageType>;
  using MaterialAttenuationsReaderType = itk::ImageFileReader<MaterialAttenuationsImageType>;

  // Read all inputs
  IncidentSpectrumReaderType::Pointer incidentSpectrumReader = IncidentSpectrumReaderType::New();
  incidentSpectrumReader->SetFileName(argv[1]);
  incidentSpectrumReader->Update();

  vIncidentSpectrumReaderType::Pointer vIncidentSpectrumReader = vIncidentSpectrumReaderType::New();
  vIncidentSpectrumReader->SetFileName(argv[2]);
  vIncidentSpectrumReader->Update();

  DetectorResponseReaderType::Pointer detectorResponseReader = DetectorResponseReaderType::New();
  detectorResponseReader->SetFileName(argv[3]);
  detectorResponseReader->Update();

  MaterialAttenuationsReaderType::Pointer materialAttenuationsReader = MaterialAttenuationsReaderType::New();
  materialAttenuationsReader->SetFileName(argv[4]);
  materialAttenuationsReader->Update();

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 4;
#else
  constexpr unsigned int NumberOfProjectionImages = 64;
#endif

  // Define the material concentrations
  MaterialPixelType concentrations;
  concentrations[0] = 0.002; // Iodine
  concentrations[1] = 0.001; // Gadolinium
  concentrations[2] = 1;     // Water

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<SingleComponentImageType>;
  ConstantImageSourceType::PointType   origin;
  ConstantImageSourceType::SizeType    size;
  ConstantImageSourceType::SpacingType spacing;

  // Generate a blank volume
  ConstantImageSourceType::Pointer tomographySource = ConstantImageSourceType::New();
#if FAST_TESTS_NO_CHECKS
  origin[0] = -64.;
  origin[1] = -64.;
  origin[2] = -64.;
  size[0] = 2;
  size[1] = 2;
  size[2] = 2;
  spacing[0] = 128.;
  spacing[1] = 128.;
  spacing[2] = 128.;
#else
  origin[0] = -124.;
  origin[1] = -124.;
  origin[2] = -124.;
  size[0] = 32;
  size[1] = 32;
  size[2] = 32;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);
  tomographySource->Update();

  // Generate a blank set of material volumes
  using MaterialVolumeSourceType = rtk::ConstantImageSource<MaterialVolumeType>;
  MaterialVolumeSourceType::Pointer materialVolumeSource = MaterialVolumeSourceType::New();
  materialVolumeSource->SetInformationFromImage(tomographySource->GetOutput());
  materialVolumeSource->SetConstant(itk::NumericTraits<MaterialPixelType>::ZeroValue());

  // Generate a blank set of projections
  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
#if FAST_TESTS_NO_CHECKS
  origin[0] = -128.;
  origin[1] = -128.;
  origin[2] = 0.;
  size[0] = 2;
  size[1] = 2;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 256.;
  spacing[1] = 256.;
  spacing[2] = 256.;
#else
  origin[0] = -240.;
  origin[1] = -240.;
  origin[2] = 0.;
  size[0] = 32;
  size[1] = 32;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 16.;
  spacing[1] = 16.;
  spacing[2] = 16.;
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);
  projectionsSource->Update();

  // Create a vectorImage of blank photon counts
  using vPhotonCountsType = itk::VectorImage<DataType, Dimension>;
  vPhotonCountsType::Pointer vPhotonCounts = vPhotonCountsType::New();
  vPhotonCounts->CopyInformation(projectionsSource->GetOutput());
  vPhotonCounts->SetVectorLength(nBins);
  vPhotonCounts->SetRegions(vPhotonCounts->GetLargestPossibleRegion());
  vPhotonCounts->Allocate();
  itk::VariableLengthVector<DataType> vZeros;
  vZeros.SetSize(nBins);
  vZeros.Fill(0);
  vPhotonCounts->FillBuffer(vZeros);

  // Geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);

  // Create ellipsoid PROJECTIONS
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<SingleComponentImageType, SingleComponentImageType>;
  REIType::Pointer rei;

  rei = REIType::New();
  REIType::VectorType semiprincipalaxis, center;
  semiprincipalaxis.Fill(90.);
  center.Fill(0.);
  rei->SetAngle(0.);
  rei->SetCenter(center);
  rei->SetAxis(semiprincipalaxis);
  rei->SetInput(projectionsSource->GetOutput());
  rei->SetGeometry(geometry);

  // Create reference volume (3D ellipsoid).
  using DEType = rtk::DrawEllipsoidImageFilter<SingleComponentImageType, SingleComponentImageType>;
  DEType::Pointer dsl = DEType::New();
  dsl->SetInput(tomographySource->GetOutput());

  // Generate one set of projections and one reference volume per material, and assemble them
  SingleComponentImageType::Pointer projs[nMaterials];
  SingleComponentImageType::Pointer vols[nMaterials];
  using ComposeProjectionsFilterType = itk::ComposeImageFilter<SingleComponentImageType>;
  ComposeProjectionsFilterType::Pointer composeProjs = ComposeProjectionsFilterType::New();
  using ComposeVolumesFilterType = itk::ComposeImageFilter<SingleComponentImageType, MaterialVolumeType>;
  ComposeVolumesFilterType::Pointer composeVols = ComposeVolumesFilterType::New();
  for (unsigned int mat = 0; mat < nMaterials; mat++)
  {
    rei->SetDensity(concentrations[mat]);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(rei->Update());
    projs[mat] = rei->GetOutput();
    projs[mat]->DisconnectPipeline();
    composeProjs->SetInput(mat, projs[mat]);

    dsl->SetDensity(concentrations[mat]);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update());
    vols[mat] = dsl->GetOutput();
    vols[mat]->DisconnectPipeline();
    composeVols->SetInput(mat, vols[mat]);
  }
  composeProjs->Update();
  composeVols->Update();

  // Apply spectral forward model to turn material projections into photon counts
  using ForwardModelFilterType =
    rtk::SpectralForwardModelImageFilter<MaterialProjectionsType, vPhotonCountsType, vIncidentSpectrum>;
  ForwardModelFilterType::Pointer forward = ForwardModelFilterType::New();
  forward->SetInputDecomposedProjections(composeProjs->GetOutput());
  forward->SetInputMeasuredProjections(vPhotonCounts);
  forward->SetInputIncidentSpectrum(vIncidentSpectrumReader->GetOutput());
  forward->SetDetectorResponse(detectorResponseReader->GetOutput());
  forward->SetMaterialAttenuations(materialAttenuationsReader->GetOutput());
  itk::VariableLengthVector<double> thresholds;
  thresholds.SetSize(nBins + 1);
  thresholds[0] = 25;
  thresholds[1] = 40;
  thresholds[2] = 55;
  thresholds[3] = 70;
  thresholds[4] = 85;
  thresholds[5] = detectorResponseReader->GetOutput()->GetLargestPossibleRegion().GetSize()[1];
  forward->SetThresholds(thresholds);
  forward->SetIsSpectralCT(true);
  forward->Update();

  // Convert the itk::VectorImage<> returned by "forward" into
  // an itk::Image<itk::Vector<>>
  using IndexSelectionType = itk::VectorIndexSelectionCastImageFilter<vPhotonCountsType, SingleComponentImageType>;
  IndexSelectionType::Pointer selectors[nBins];
  using ComposePhotonCountsType = itk::ComposeImageFilter<SingleComponentImageType, PhotonCountsType>;
  ComposePhotonCountsType::Pointer composePhotonCounts = ComposePhotonCountsType::New();
  for (unsigned int bin = 0; bin < nBins; bin++)
  {
    selectors[bin] = IndexSelectionType::New();
    selectors[bin]->SetIndex(bin);
    selectors[bin]->SetInput(forward->GetOutput());
    composePhotonCounts->SetInput(bin, selectors[bin]->GetOutput());
  }
  composePhotonCounts->Update();

  // Read the material attenuations image as a matrix
  MaterialAttenuationsImageType::IndexType indexMat;
  vnl_matrix<DataType>                     materialAttenuationsMatrix(nEnergies, nMaterials);
  for (unsigned int energy = 0; energy < nEnergies; energy++)
  {
    indexMat[1] = energy;
    for (unsigned int material = 0; material < nMaterials; material++)
    {
      indexMat[0] = material;
      materialAttenuationsMatrix[energy][material] = materialAttenuationsReader->GetOutput()->GetPixel(indexMat);
    }
  }

  // Read the detector response image as a matrix, and bin it
  vnl_matrix<DataType> drm =
    rtk::SpectralBinDetectorResponse<DataType>(detectorResponseReader->GetOutput(), thresholds, nEnergies);

  // Reconstruct using Mechlem
  using MechlemType =
    rtk::MechlemOneStepSpectralReconstructionFilter<MaterialVolumeType, PhotonCountsType, IncidentSpectrumImageType>;
  MechlemType::Pointer mechlemOneStep = MechlemType::New();
  mechlemOneStep->SetForwardProjectionFilter(MechlemType::FP_JOSEPH); // Joseph
  mechlemOneStep->SetInputMaterialVolumes(materialVolumeSource->GetOutput());
  mechlemOneStep->SetInputPhotonCounts(composePhotonCounts->GetOutput());
  mechlemOneStep->SetInputSpectrum(incidentSpectrumReader->GetOutput());
  mechlemOneStep->SetBinnedDetectorResponse(drm);
  mechlemOneStep->SetMaterialAttenuations(materialAttenuationsMatrix);
  mechlemOneStep->SetGeometry(geometry);
  mechlemOneStep->SetNumberOfIterations(20);

  std::cout << "\n\n****** Case 1: Joseph Backprojector ******" << std::endl;

  mechlemOneStep->SetBackProjectionFilter(MechlemType::BP_JOSEPH);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(mechlemOneStep->Update());

  CheckVectorImageQuality<MaterialVolumeType>(mechlemOneStep->GetOutput(), composeVols->GetOutput(), 0.08, 22, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Voxel-based Backprojector ******" << std::endl;

  mechlemOneStep->SetBackProjectionFilter(MechlemType::BP_VOXELBASED);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(mechlemOneStep->Update());

  CheckVectorImageQuality<MaterialVolumeType>(mechlemOneStep->GetOutput(), composeVols->GetOutput(), 0.08, 22, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 3: Voxel-based Backprojector, 4 subsets, with regularization  ******" << std::endl;

  using ReorderProjectionsFilterType = rtk::ReorderProjectionsImageFilter<PhotonCountsType>;
  ReorderProjectionsFilterType::Pointer reorder = ReorderProjectionsFilterType::New();
  reorder->SetInput(composePhotonCounts->GetOutput());
  reorder->SetInputGeometry(geometry);
  reorder->SetPermutation(rtk::ReorderProjectionsImageFilter<PhotonCountsType>::SHUFFLE);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reorder->Update())
  mechlemOneStep->SetInputPhotonCounts(reorder->GetOutput());
  mechlemOneStep->SetGeometry(reorder->GetOutputGeometry());

  mechlemOneStep->SetBackProjectionFilter(MechlemType::BP_VOXELBASED);
  mechlemOneStep->SetNumberOfSubsets(4);
  mechlemOneStep->SetNumberOfIterations(5);
  MaterialVolumeType::RegionType::SizeType radius;
  radius.Fill(1);
  MaterialVolumeType::PixelType weights;
  weights.Fill(1);
  mechlemOneStep->SetRegularizationRadius(radius);
  mechlemOneStep->SetRegularizationWeights(weights);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(mechlemOneStep->Update());

  CheckVectorImageQuality<MaterialVolumeType>(mechlemOneStep->GetOutput(), composeVols->GetOutput(), 0.08, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef RTK_USE_CUDA
  std::cout << "\n\n****** Case 4: CUDA voxel-based Backprojector, 4 subsets, with regularization ******" << std::endl;

  mechlemOneStep->SetForwardProjectionFilter(MechlemType::FP_CUDARAYCAST);
  mechlemOneStep->SetBackProjectionFilter(MechlemType::BP_CUDAVOXELBASED);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(mechlemOneStep->Update());

  CheckVectorImageQuality<MaterialVolumeType>(mechlemOneStep->GetOutput(), composeVols->GetOutput(), 0.08, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  return EXIT_SUCCESS;
}
