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
 * This test reads decomposed projections, measured projections, incident spectrum, material
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
    std::cerr
      << argv[0]
      << " decomposedProjections.mha measuredProjections.mha incidentSpectrum.mha projections.mha DetectorResponse.csv "
         "materialAttenuations.csv out1.mha out2.mha"
      << std::endl;
    return EXIT_FAILURE;
  }

  // Define types
  constexpr unsigned int nBins = 5;
  constexpr unsigned int nMaterials = 3;
  constexpr unsigned int nEnergies = 150;
  using dataType = double;
  using TDecomposedProjections = itk::Image<itk::Vector<dataType, nMaterials>, 3>;
  using TMeasuredProjections = itk::Image<itk::Vector<dataType, nBins>, 3>;
  using TIncidentSpectrum = itk::Image<dataType, 3>;
  using TProjections = itk::Image<dataType, 3>;
  using TOutput1 = itk::Image<itk::Vector<dataType, nMaterials>, 3>;
  using TOutput2 = itk::Image<itk::Vector<dataType, nMaterials * nMaterials>, 3>;

  vnl_matrix<dataType> detectorResponse(nBins, nEnergies);
  vnl_matrix<dataType> materialAttenuations(nEnergies, nMaterials);

  // Define, instantiate, set and update readers
  using DecomposedProjectionsReaderType = itk::ImageFileReader<TDecomposedProjections>;
  auto decomposedProjectionsReader = DecomposedProjectionsReaderType::New();
  decomposedProjectionsReader->SetFileName(argv[1]);
  decomposedProjectionsReader->Update();

  using MeasuredProjectionsReaderType = itk::ImageFileReader<TMeasuredProjections>;
  auto measuredProjectionsReader = MeasuredProjectionsReaderType::New();
  measuredProjectionsReader->SetFileName(argv[2]);
  measuredProjectionsReader->Update();

  using IncidentSpectrumReaderType = itk::ImageFileReader<TIncidentSpectrum>;
  auto incidentSpectrumReader = IncidentSpectrumReaderType::New();
  incidentSpectrumReader->SetFileName(argv[3]);
  incidentSpectrumReader->Update();

  using ProjectionsReaderType = itk::ImageFileReader<TProjections>;
  auto projectionsReader = ProjectionsReaderType::New();
  projectionsReader->SetFileName(argv[4]);
  projectionsReader->Update();

  using Output1ReaderType = itk::ImageFileReader<TOutput1>;
  auto output1Reader = Output1ReaderType::New();
  output1Reader->SetFileName(argv[7]);
  output1Reader->Update();

  using Output2ReaderType = itk::ImageFileReader<TOutput2>;
  auto output2Reader = Output2ReaderType::New();
  output2Reader->SetFileName(argv[8]);
  output2Reader->Update();

  // Read binned detector response
  using CSVReaderType = itk::CSVArray2DFileReader<dataType>;
  auto csvReader = CSVReaderType::New();
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
  using WeidingerForwardModelType = rtk::
    WeidingerForwardModelImageFilter<TDecomposedProjections, TMeasuredProjections, TIncidentSpectrum, TProjections>;
  auto weidingerForward = WeidingerForwardModelType::New();

  // Set its inputs
  weidingerForward->SetInputDecomposedProjections(decomposedProjectionsReader->GetOutput());
  weidingerForward->SetInputMeasuredProjections(measuredProjectionsReader->GetOutput());
  weidingerForward->SetInputIncidentSpectrum(incidentSpectrumReader->GetOutput());
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
