#include "rtkTest.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkMechlemOneStepSpectralReconstructionFilter.h"
#include "rtkSpectralForwardModelImageFilter.h"

//#ifdef USE_CUDA
//  #include "itkCudaImage.h"
//#endif

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

int main(int, char** )
{
  const unsigned int Dimension = 3;
  const unsigned int nBins = 5;
  const unsigned int nMaterials = 3;
  const unsigned int nEnergies = 150;
  typedef float                               DataType;
  typedef itk::Vector<DataType, nMaterials>   MaterialPixelType;
  typedef itk::Vector<DataType, nBins>        PhotonCountsPixelType;
  typedef itk::Vector<DataType, nEnergies>    SpectrumPixelType;
#ifdef USE_CUDA
#else
  typedef itk::Image<MaterialPixelType, Dimension>        MaterialVolumeType;
  typedef itk::VectorImage<DataType, Dimension>           MaterialProjectionsType;
  typedef itk::Image<PhotonCountsPixelType, Dimension>    PhotonCountsType;
  typedef itk::Image<SpectrumPixelType, Dimension-1 >     IncidentSpectrumImageType;
  typedef itk::VectorImage<DataType, Dimension - 1>       vIncidentSpectrum;
  typedef itk::Image<DataType, Dimension-1 >              DetectorResponseImageType;
  typedef itk::Image<DataType, Dimension-1 >              MaterialAttenuationsImageType;
  typedef itk::Image<DataType, Dimension>                 SingleComponentImageType;
#endif

  typedef itk::ImageFileReader<IncidentSpectrumImageType> IncidentSpectrumReaderType;
  typedef itk::ImageFileReader<vIncidentSpectrum> vIncidentSpectrumReaderType;
  typedef itk::ImageFileReader<DetectorResponseImageType> DetectorResponseReaderType;
  typedef itk::ImageFileReader<MaterialAttenuationsImageType> MaterialAttenuationsReaderType;

  // Read all inputs
  IncidentSpectrumReaderType::Pointer incidentSpectrumReader = IncidentSpectrumReaderType::New();
  incidentSpectrumReader->SetFileName( std::string(RTK_DATA_ROOT) +
                                       std::string("/Input/Spectral/OneStep/incident_spectrum_64_rows.mha") );
  incidentSpectrumReader->Update();

  vIncidentSpectrumReaderType::Pointer vIncidentSpectrumReader = vIncidentSpectrumReaderType::New();
  vIncidentSpectrumReader->SetFileName( std::string(RTK_DATA_ROOT) +
                                       std::string("/Input/Spectral/OneStep/incident_spectrum_64_rows.mha") );
  vIncidentSpectrumReader->Update();

  DetectorResponseReaderType::Pointer detectorResponseReader = DetectorResponseReaderType::New();
  detectorResponseReader->SetFileName( std::string(RTK_DATA_ROOT) +
                                       std::string("/Input/Spectral/detector_response.mha") );
  detectorResponseReader->Update();

  MaterialAttenuationsReaderType::Pointer materialAttenuationsReader = MaterialAttenuationsReaderType::New();
  materialAttenuationsReader->SetFileName( std::string(RTK_DATA_ROOT) +
                                           std::string("/Input/Spectral/material_attenuations.mha") );
  materialAttenuationsReader->Update();

#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfProjectionImages = 1;
#else
  const unsigned int NumberOfProjectionImages = 64;
#endif

  // Define the material concentrations
  MaterialPixelType concentrations;
  concentrations[0] = 0.002; // Iodine
  concentrations[1] = 0.001; // Gadolinium
  concentrations[2] = 1;    // Water

  // Constant image sources
  typedef rtk::ConstantImageSource< SingleComponentImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  // Generate a blank volume
  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = 2;
  spacing[0] = 252.;
  spacing[1] = 252.;
  spacing[2] = 252.;
#else
  size[0] = 64;
  size[1] = 64;
  size[2] = 64;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
#endif
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );
  tomographySource->Update();

  // Generate a blank set of material volumes
  typedef rtk::ConstantImageSource< MaterialVolumeType > MaterialVolumeSourceType;
  MaterialVolumeSourceType::Pointer materialVolumeSource = MaterialVolumeSourceType::New();
  materialVolumeSource->SetInformationFromImage(tomographySource->GetOutput());
  materialVolumeSource->SetConstant( itk::NumericTraits<MaterialPixelType>::Zero );

  // Generate a blank set of projections
  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -255.;
  origin[1] = -255.;
  origin[2] = -255.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 504.;
  spacing[1] = 504.;
  spacing[2] = 504.;
#else
  size[0] = 64;
  size[1] = 64;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
#endif
  projectionsSource->SetOrigin( origin );
  projectionsSource->SetSpacing( spacing );
  projectionsSource->SetSize( size );
  projectionsSource->SetConstant( 0. );
  projectionsSource->Update();

  // Create a vectorImage of blank photon counts
  typedef itk::VectorImage<DataType, Dimension> vPhotonCountsType;
  typename vPhotonCountsType::Pointer vPhotonCounts = vPhotonCountsType::New();
  vPhotonCounts->CopyInformation(projectionsSource->GetOutput());
  vPhotonCounts->SetVectorLength(nBins);
  vPhotonCounts->SetRegions(vPhotonCounts->GetLargestPossibleRegion());
  vPhotonCounts->Allocate();
  itk::VariableLengthVector<DataType> vZeros;
  vZeros.SetSize(nBins);
  vZeros.Fill(0);
  vPhotonCounts->FillBuffer(vZeros);

  // Geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages);

  // Create ellipsoid PROJECTIONS
  typedef rtk::RayEllipsoidIntersectionImageFilter<SingleComponentImageType, SingleComponentImageType> REIType;
  REIType::Pointer rei;

  rei = REIType::New();
  REIType::VectorType semiprincipalaxis, center;
  semiprincipalaxis.Fill(90.);
  center.Fill(0.);
  rei->SetAngle(0.);
  rei->SetCenter(center);
  rei->SetAxis(semiprincipalaxis);
  rei->SetInput( projectionsSource->GetOutput() );
  rei->SetGeometry( geometry );

  // Create reference volume (3D ellipsoid).
  typedef rtk::DrawEllipsoidImageFilter<SingleComponentImageType, SingleComponentImageType> DEType;
  DEType::Pointer dsl = DEType::New();
  dsl->SetInput( tomographySource->GetOutput() );

  // Generate one set of projections and one reference volume per material, and assemble them
  SingleComponentImageType::Pointer projs[nMaterials];
  SingleComponentImageType::Pointer vols[nMaterials];
  typedef itk::ComposeImageFilter<SingleComponentImageType> ComposeProjectionsFilterType;
  ComposeProjectionsFilterType::Pointer composeProjs = ComposeProjectionsFilterType::New();
  typedef itk::ComposeImageFilter<SingleComponentImageType, MaterialVolumeType> ComposeVolumesFilterType;
  ComposeVolumesFilterType::Pointer composeVols = ComposeVolumesFilterType::New();
  for (unsigned int mat=0; mat<nMaterials; mat++)
    {
    rei->SetDensity(concentrations[mat]);
    TRY_AND_EXIT_ON_ITK_EXCEPTION( rei->Update() );
    projs[mat] = rei->GetOutput();
    projs[mat]->DisconnectPipeline();
    composeProjs->SetInput(mat, projs[mat]);

    dsl->SetDensity(concentrations[mat]);
    TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() );
    vols[mat] = dsl->GetOutput();
    vols[mat]->DisconnectPipeline();
    composeVols->SetInput(mat, vols[mat]);
    }
  composeProjs->Update();
  composeVols->Update();

  // Apply spectral forward model to turn material projections into photon counts
  typedef rtk::SpectralForwardModelImageFilter<MaterialProjectionsType, vPhotonCountsType, vIncidentSpectrum> ForwardModelFilterType;
  ForwardModelFilterType::Pointer forward = ForwardModelFilterType::New();
  forward->SetInputDecomposedProjections(composeProjs->GetOutput());
  forward->SetInputMeasuredProjections(vPhotonCounts);
  forward->SetInputIncidentSpectrum(vIncidentSpectrumReader->GetOutput());
  forward->SetDetectorResponse(detectorResponseReader->GetOutput());
  forward->SetMaterialAttenuations(materialAttenuationsReader->GetOutput());
  itk::VariableLengthVector<unsigned int> thresholds;
  thresholds.SetSize(nBins+1);
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
  typedef itk::VectorIndexSelectionCastImageFilter<vPhotonCountsType, SingleComponentImageType> IndexSelectionType;
  IndexSelectionType::Pointer selectors[nBins];
  typedef itk::ComposeImageFilter<SingleComponentImageType, PhotonCountsType> ComposePhotonCountsType;
  ComposePhotonCountsType::Pointer composePhotonCounts = ComposePhotonCountsType::New();
  for (unsigned int bin=0; bin<nBins; bin++)
    {
    selectors[bin] = IndexSelectionType::New();
    selectors[bin]->SetIndex(bin);
    selectors[bin]->SetInput(forward->GetOutput());
    composePhotonCounts->SetInput(bin, selectors[bin]->GetOutput());
    }
  composePhotonCounts->Update();

  // Read the material attenuations image as a matrix
  typename MaterialAttenuationsImageType::IndexType indexMat;
  itk::Matrix<DataType, nEnergies, nMaterials> materialAttenuationsMatrix;
  for (unsigned int energy=0; energy<nEnergies; energy++)
    {
    indexMat[1] = energy;
    for (unsigned int material=0; material<nMaterials; material++)
      {
      indexMat[0] = material;
      materialAttenuationsMatrix[energy][material] = materialAttenuationsReader->GetOutput()->GetPixel(indexMat);
      }
    }

  // Read the detector response image as a matrix, and bin it
  itk::Matrix<DataType, nBins, nEnergies> detectorResponseMatrix;
  typename DetectorResponseImageType::IndexType indexDet;
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

  // Reconstruct using Mechlem
  typedef rtk::MechlemOneStepSpectralReconstructionFilter<MaterialVolumeType, PhotonCountsType, IncidentSpectrumImageType> MechlemType;
  MechlemType::Pointer mechlemOneStep = MechlemType::New();
  mechlemOneStep->SetForwardProjectionFilter( 0 ); // Joseph
  mechlemOneStep->SetInputMaterialVolumes( materialVolumeSource->GetOutput() );
  mechlemOneStep->SetInputPhotonCounts(composePhotonCounts->GetOutput());
  mechlemOneStep->SetInputSpectrum(incidentSpectrumReader->GetOutput());
  mechlemOneStep->SetBinnedDetectorResponse(detectorResponseMatrix);
  mechlemOneStep->SetMaterialAttenuations(materialAttenuationsMatrix);
  mechlemOneStep->SetGeometry( geometry );
  mechlemOneStep->SetNumberOfIterations( 20 );

  std::cout << "\n\n****** Case 1: Joseph Backprojector ******" << std::endl;

  mechlemOneStep->SetBackProjectionFilter( 1 );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( mechlemOneStep->Update() );

  CheckVectorImageQuality<MaterialVolumeType>(mechlemOneStep->GetOutput(), composeVols->GetOutput(), 0.08, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Voxel-based Backprojector ******" << std::endl;

  mechlemOneStep->SetBackProjectionFilter( 0 );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( mechlemOneStep->Update() );

  CheckVectorImageQuality<MaterialVolumeType>(mechlemOneStep->GetOutput(), composeVols->GetOutput(), 0.08, 23, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;


#ifdef USE_CUDA
#endif

  return EXIT_SUCCESS;
}
