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

int
main(int argc, char * argv[])
{
  if (argc < 9)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0]
              << " materialProjections.mha photonCounts.mha spectrum.mha projections.mha DetectorResponse.csv "
                 "materialAttenuations.csv out1.mha out2.mha"
              << std::endl;
    return EXIT_FAILURE;
  }

  // Define types
  constexpr unsigned int nBins = 5;
  constexpr unsigned int nMaterials = 3;
  constexpr unsigned int nEnergies = 150;
  using dataType = double;
  using TMaterialProjections = itk::Image<itk::Vector<dataType, nMaterials>, 3>;
  using TPhotonCounts = itk::Image<itk::Vector<dataType, nBins>, 3>;
  using TSpectrum = itk::Image<dataType, 3>;
  using TProjections = itk::Image<dataType, 3>;
  using TOutput1 = itk::Image<itk::Vector<dataType, nMaterials>, 3>;
  using TOutput2 = itk::Image<itk::Vector<dataType, nMaterials * nMaterials>, 3>;

  vnl_matrix<dataType> detectorResponse(nBins, nEnergies);
  vnl_matrix<dataType> materialAttenuations(nEnergies, nMaterials);

  // Define, instantiate, set and update readers
  using MaterialProjectionsReaderType = itk::ImageFileReader<TMaterialProjections>;
  MaterialProjectionsReaderType::Pointer materialProjectionsReader = MaterialProjectionsReaderType::New();
  materialProjectionsReader->SetFileName(argv[1]);
  materialProjectionsReader->Update();

  using PhotonCountsReaderType = itk::ImageFileReader<TPhotonCounts>;
  PhotonCountsReaderType::Pointer photonCountsReader = PhotonCountsReaderType::New();
  photonCountsReader->SetFileName(argv[2]);
  photonCountsReader->Update();

  using SpectrumReaderType = itk::ImageFileReader<TSpectrum>;
  SpectrumReaderType::Pointer spectrumReader = SpectrumReaderType::New();
  spectrumReader->SetFileName(argv[3]);
  spectrumReader->Update();

  using ProjectionsReaderType = itk::ImageFileReader<TProjections>;
  ProjectionsReaderType::Pointer projectionsReader = ProjectionsReaderType::New();
  projectionsReader->SetFileName(argv[4]);
  projectionsReader->Update();

  using Output1ReaderType = itk::ImageFileReader<TOutput1>;
  Output1ReaderType::Pointer output1Reader = Output1ReaderType::New();
  output1Reader->SetFileName(argv[7]);
  output1Reader->Update();

  using Output2ReaderType = itk::ImageFileReader<TOutput2>;
  Output2ReaderType::Pointer output2Reader = Output2ReaderType::New();
  output2Reader->SetFileName(argv[8]);
  output2Reader->Update();

  // Read binned detector response
  using CSVReaderType = itk::CSVArray2DFileReader<dataType>;
  CSVReaderType::Pointer csvReader = CSVReaderType::New();
  csvReader->SetFieldDelimiterCharacter(',');
  csvReader->HasColumnHeadersOff();
  csvReader->HasRowHeadersOff();

  std::string filename = argv[5];
  csvReader->SetFileName(filename);
  csvReader->Parse();
  for (unsigned int r = 0; r < nBins; r++)
    for (unsigned int c = 0; c < nEnergies; c++)
      detectorResponse[r][c] = csvReader->GetOutput()->GetData(r, c);

  // Read material attenuations
  filename = argv[6];
  csvReader->SetFileName(filename);
  csvReader->Parse();
  for (unsigned int r = 0; r < nEnergies; r++)
    for (unsigned int c = 0; c < nMaterials; c++)
      materialAttenuations[r][c] = csvReader->GetOutput()->GetData(r, c);

  // Create the filter
  using WeidingerForwardModelType =
    rtk::WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>;
  WeidingerForwardModelType::Pointer weidingerForward = WeidingerForwardModelType::New();

  // Set its inputs
  weidingerForward->SetInputMaterialProjections(materialProjectionsReader->GetOutput());
  weidingerForward->SetInputPhotonCounts(photonCountsReader->GetOutput());
  weidingerForward->SetInputSpectrum(spectrumReader->GetOutput());
  weidingerForward->SetInputProjectionsOfOnes(projectionsReader->GetOutput());
  weidingerForward->SetBinnedDetectorResponse(detectorResponse);
  weidingerForward->SetMaterialAttenuations(materialAttenuations);

  // Update the filter
  TRY_AND_EXIT_ON_ITK_EXCEPTION(weidingerForward->Update());

  // 2. Compare read projections
  CheckVectorImageQuality<TOutput1>(weidingerForward->GetOutput1(), output1Reader->GetOutput(), 1.e-9, 200, 2000.0);
  CheckVectorImageQuality<TOutput2>(weidingerForward->GetOutput2(), output2Reader->GetOutput(), 1.e-7, 200, 2000.0);

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
