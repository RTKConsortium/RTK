#include "rtkTest.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "rtkSimplexSpectralProjectionsDecompositionImageFilter.h"
#include <itkImageFileReader.h>

/**
 * \file rtkdecomposespectralprojectionstest.cxx
 *
 * \brief Functional test for the filter performing spectral projections' material decomposition
 *
 * This test performs a material decomposition on a set of 4 projections of size 16x8.
 *
 * \author Cyril Mory
 */

int main(int , char** )
{
  typedef float PixelValueType;
  const unsigned int Dimension = 3;
  const unsigned int NumberOfMaterials = 3;
  const unsigned int NumberOfSpectralBins = 6;
  const unsigned int MaximumEnergy = 150;

  typedef itk::Vector<PixelValueType, NumberOfMaterials> MaterialsVectorType;
  typedef itk::Image< MaterialsVectorType, Dimension > DecomposedProjectionType;
  typedef itk::ImageFileReader<DecomposedProjectionType> DecomposedProjectionReaderType;

  typedef itk::Vector<PixelValueType, NumberOfSpectralBins> SpectralVectorType;
  typedef itk::Image< SpectralVectorType, Dimension > SpectralProjectionsType;
  typedef itk::ImageFileReader< SpectralProjectionsType > SpectralProjectionReaderType;

  typedef itk::Vector<PixelValueType, MaximumEnergy> IncidentSpectrumVectorType;
  typedef itk::Image< IncidentSpectrumVectorType, Dimension-1 > IncidentSpectrumImageType;
  typedef itk::ImageFileReader<IncidentSpectrumImageType> IncidentSpectrumReaderType;

  typedef itk::Image< PixelValueType, Dimension-1 > DetectorResponseImageType;
  typedef itk::ImageFileReader<DetectorResponseImageType> DetectorResponseReaderType;

  typedef itk::Image< PixelValueType, Dimension-1 > MaterialAttenuationsImageType;
  typedef itk::ImageFileReader<MaterialAttenuationsImageType> MaterialAttenuationsReaderType;

  typedef itk::Matrix<PixelValueType, NumberOfSpectralBins, MaximumEnergy>            DetectorResponseType;
  typedef itk::Vector<itk::Vector<PixelValueType, MaximumEnergy>, NumberOfMaterials>  MaterialAttenuationsType;

  // Generate an initial set of decomposed projections
  DecomposedProjectionType::Pointer initialDecomposedProjections = DecomposedProjectionType::New();
  DecomposedProjectionType::SizeType initSize;
  DecomposedProjectionType::IndexType initIndex;
  initSize[0]=16;
  initSize[1]=8;
  initSize[2]=4;
  initIndex.Fill(0);
  DecomposedProjectionType::RegionType initRegion;
  initRegion.SetSize(initSize);
  initRegion.SetIndex(initIndex);
  initialDecomposedProjections->SetRegions(initRegion);
  initialDecomposedProjections->Allocate();
  DecomposedProjectionType::PixelType initPixel;
  initPixel[0] = 0.1;
  initPixel[1] = 0.1;
  initPixel[2] = 100;
  initialDecomposedProjections->FillBuffer(initPixel);

  // Read all inputs
  SpectralProjectionReaderType::Pointer spectralProjectionReader = SpectralProjectionReaderType::New();
  spectralProjectionReader->SetFileName( std::string(RTK_DATA_ROOT) +
                                         std::string("/Input/Spectral/photon_count.mha") );
  spectralProjectionReader->Update();

  IncidentSpectrumReaderType::Pointer incidentSpectrumReader = IncidentSpectrumReaderType::New();
  incidentSpectrumReader->SetFileName( std::string(RTK_DATA_ROOT) +
                                       std::string("/Input/Spectral/incident_spectrum.mha") );
  incidentSpectrumReader->Update();

  DetectorResponseReaderType::Pointer detectorResponseReader = DetectorResponseReaderType::New();
  detectorResponseReader->SetFileName( std::string(RTK_DATA_ROOT) +
                                       std::string("/Input/Spectral/detector_response.mha") );
  detectorResponseReader->Update();

  MaterialAttenuationsReaderType::Pointer materialAttenuationsReader = MaterialAttenuationsReaderType::New();
  materialAttenuationsReader->SetFileName( std::string(RTK_DATA_ROOT) +
                                           std::string("/Input/Spectral/material_attenuations.mha") );
  materialAttenuationsReader->Update();

  // Read baseline output
  DecomposedProjectionReaderType::Pointer decomposedProjectionReader = DecomposedProjectionReaderType::New();
  decomposedProjectionReader->SetFileName( std::string(RTK_DATA_ROOT) +
                                           std::string("/Baseline/Spectral/ref_output.mha") );
  decomposedProjectionReader->Update();

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
  simplex->SetInputDecomposedProjections(initialDecomposedProjections);
  simplex->SetInputSpectralProjections(spectralProjectionReader->GetOutput());
  simplex->SetInputIncidentSpectrum(incidentSpectrumReader->GetOutput());
  simplex->SetDetectorResponse(detectorResponseMatrix);
  simplex->SetMaterialAttenuations(materialAttenuationsVector);
  simplex->SetNumberOfIterations(10000);
  simplex->SetNumberOfThreads(1);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(simplex->Update())

  CheckVectorImageQuality<DecomposedProjectionType>(simplex->GetOutput(), decomposedProjectionReader->GetOutput(), 0.00001, 15, 2.0);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
