#include "rtkTest.h"
#include "rtkMacro.h"
#include <itkImageFileReader.h>
#include "rtkWeidingerForwardModelImageFilter.h"
#include <itkCSVArray2DFileReader.h>

/**
 * \file rtkweidingerforwardmodeltest.cxx
 *
 * \brief Test for the filter rtkWeidingerForwardModelImageFilter
 *
 * This test reads material projections, photon counts, spectrum, material
 * attenuations, detector response, and projections of a volume of ones,
 * runs the filter rtkWeidingerForwardModelImageFilter, and compare its outputs
 * to the expected ones (computed with Matlab)
 *
 * \author Cyril Mory
 */

int main(int, char** )
{
  // Define types
  const unsigned int nBins=5;
  const unsigned int nMaterials=3;
  const unsigned int nEnergies=150;
  typedef double dataType;
  typedef itk::Image<itk::Vector<dataType, nMaterials>, 3> TMaterialProjections;
  typedef itk::Image<itk::Vector<dataType, nBins>, 3> TPhotonCounts;
  typedef itk::Image<itk::Vector<dataType, nEnergies>, 3> TSpectrum;
  typedef itk::Image<dataType, 3> TProjections;
  typedef itk::Image<itk::Vector<dataType, nMaterials>, 3> TOutput1;
  typedef itk::Image<itk::Vector<dataType, nMaterials * nMaterials>, 3> TOutput2;

  typedef itk::Matrix<dataType, nBins, nEnergies>       BinnedDetectorResponseType;
  typedef itk::Matrix<dataType, nEnergies, nMaterials>  MaterialAttenuationsType;

  // Define, instantiate, set and update readers
  typedef itk::ImageFileReader<TMaterialProjections> MaterialProjectionsReaderType;
  MaterialProjectionsReaderType::Pointer materialProjectionsReader = MaterialProjectionsReaderType::New();
  materialProjectionsReader->SetFileName(std::string(RTK_DATA_ROOT) + std::string("/Input/Spectral/OneStep/materialProjections.mha"));
  materialProjectionsReader->Update();

  typedef itk::ImageFileReader<TPhotonCounts> PhotonCountsReaderType;
  PhotonCountsReaderType::Pointer photonCountsReader = PhotonCountsReaderType::New();
  photonCountsReader->SetFileName(std::string(RTK_DATA_ROOT) + std::string("/Input/Spectral/OneStep/photonCounts.mha"));
  photonCountsReader->Update();

  typedef itk::ImageFileReader<TSpectrum> SpectrumReaderType;
  SpectrumReaderType::Pointer spectrumReader = SpectrumReaderType::New();
  spectrumReader->SetFileName(std::string(RTK_DATA_ROOT) + std::string("/Input/Spectral/OneStep/spectrum.mha"));
  spectrumReader->Update();

  typedef itk::ImageFileReader<TProjections> ProjectionsReaderType;
  ProjectionsReaderType::Pointer projectionsReader = ProjectionsReaderType::New();
  projectionsReader->SetFileName(std::string(RTK_DATA_ROOT) + std::string("/Input/Spectral/OneStep/projOfOnes.mha"));
  projectionsReader->Update();

  typedef itk::ImageFileReader<TOutput1> Output1ReaderType;
  Output1ReaderType::Pointer output1Reader = Output1ReaderType::New();
  output1Reader->SetFileName(std::string(RTK_DATA_ROOT) + std::string("/Baseline/Spectral/OneStep/out1.mha"));
  output1Reader->Update();

  typedef itk::ImageFileReader<TOutput2> Output2ReaderType;
  Output2ReaderType::Pointer output2Reader = Output2ReaderType::New();
  output2Reader->SetFileName(std::string(RTK_DATA_ROOT) + std::string("/Baseline/Spectral/OneStep/out2.mha"));
  output2Reader->Update();

  // Read binned detector response
  typedef itk::CSVArray2DFileReader<dataType> CSVReaderType;
  CSVReaderType::Pointer csvReader = CSVReaderType::New();
  csvReader->SetFieldDelimiterCharacter( ',' );
  csvReader->HasColumnHeadersOff();
  csvReader->HasRowHeadersOff();

  std::string filename = std::string(RTK_DATA_ROOT) + std::string("/Input/Spectral/OneStep/binnedDetectorResponse.csv");
  csvReader->SetFileName( filename );
  csvReader->Parse();
  BinnedDetectorResponseType detectorResponse;
  for (unsigned int r=0; r<nBins; r++)
    for (unsigned int c=0; c<nEnergies; c++)
      detectorResponse[r][c] = csvReader->GetOutput()->GetData(r, c);

  // Read material attenuations
  filename = std::string(RTK_DATA_ROOT) + std::string("/Input/Spectral/OneStep/materialAttenuations.csv");
  csvReader->SetFileName( filename );
  csvReader->Parse();
  MaterialAttenuationsType materialAttenuations;
  for (unsigned int r=0; r<nEnergies; r++)
    for (unsigned int c=0; c<nMaterials; c++)
      materialAttenuations[r][c] = csvReader->GetOutput()->GetData(r, c);

  // Create the filter
  typedef rtk::WeidingerForwardModelImageFilter< TMaterialProjections,
                                                 TPhotonCounts,
                                                 TSpectrum,
                                                 TProjections> WeidingerForwardModelType;
  WeidingerForwardModelType::Pointer weidingerForward = WeidingerForwardModelType::New();

  // Set its inputs
  weidingerForward->SetInputMaterialProjections(materialProjectionsReader->GetOutput());
  weidingerForward->SetInputPhotonCounts(photonCountsReader->GetOutput());
  weidingerForward->SetInputSpectrum(spectrumReader->GetOutput());
  weidingerForward->SetInputProjectionsOfOnes(projectionsReader->GetOutput());
  weidingerForward->SetBinnedDetectorResponse(detectorResponse);
  weidingerForward->SetMaterialAttenuations(materialAttenuations);

  // Update the filter
  TRY_AND_EXIT_ON_ITK_EXCEPTION( weidingerForward->Update() );

  // 2. Compare read projections
  CheckVectorImageQuality< TOutput1 >(weidingerForward->GetOutput1(), output1Reader->GetOutput(), 1.e-9, 200, 2000.0);
  CheckVectorImageQuality< TOutput2 >(weidingerForward->GetOutput2(), output2Reader->GetOutput(), 1.e-7, 200, 2000.0);

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
