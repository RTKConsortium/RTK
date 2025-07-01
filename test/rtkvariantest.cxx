#include "rtkTest.h"
#include "rtkProjectionsReader.h"
#include "rtkMacro.h"
#include "rtkVarianObiGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkVarianProBeamGeometryReader.h"

#include <itkRegularExpressionSeriesFileNames.h>

/**
 * \file rtkvariantest.cxx
 *
 * \brief Functional tests for classes managing Varian data
 *
 * This test reads a projection and the geometry of an acquisition from a
 * Varian acquisition and compares it to the expected results, which are
 * read from a baseline image in the MetaIO file format and a geometry file in
 * the RTK format, respectively.
 *
 * \author Simon Rit
 */

int
main(int argc, char * argv[])
{
  if (argc < 12)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  projection.hnd acqui.xml"
              << " ximFile.xim varianGeometry.xml"
              << " projection.hnc"
              << " reference.xml refGeometry.mha"
              << " referenceXim.xml refGeometryXim.mha"
              << " referenceHnc.xml refGeometryHnc.mha" << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<std::string> fileNames;
  fileNames.emplace_back(argv[1]);

  // Varian geometry
  rtk::VarianObiGeometryReader::Pointer geoTargReader;
  geoTargReader = rtk::VarianObiGeometryReader::New();
  geoTargReader->SetXMLFileName(argv[2]);
  geoTargReader->SetProjectionsFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoTargReader->UpdateOutputData());

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename(argv[6]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoRefReader->GenerateOutputInformation())

  // 1. Check geometries
  CheckGeometries(geoTargReader->GetGeometry(), geoRefReader->GetOutputObject());

  // ******* COMPARING projections *******
  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;
  using ImageType = itk::Image<OutputPixelType, Dimension>;

  // Varian projections reader
  using ReaderType = rtk::ProjectionsReader<ImageType>;
  auto reader = ReaderType::New();
  reader->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update());

  // Reference projections reader
  auto readerRef = ReaderType::New();
  fileNames.clear();
  fileNames.emplace_back(argv[7]);
  readerRef->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality<ImageType>(reader->GetOutput(), readerRef->GetOutput(), 1e-8, 100, 2.0);

  ///////////////////// Xim file format
  fileNames.clear();
  fileNames.emplace_back(argv[3]);

  // Varian geometry
  rtk::VarianProBeamGeometryReader::Pointer geoProBeamReader;
  geoProBeamReader = rtk::VarianProBeamGeometryReader::New();
  geoProBeamReader->SetXMLFileName(argv[4]);
  geoProBeamReader->SetProjectionsFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoProBeamReader->UpdateOutputData());

  // Reference geometry
  geoRefReader->SetFilename(argv[8]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoRefReader->GenerateOutputInformation())

  // 1. Check geometries
  CheckGeometries(geoProBeamReader->GetGeometry(), geoRefReader->GetOutputObject());

  // ******* COMPARING projections *******
  // Varian projections reader
  reader->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update());

  // Reference projections reader
  fileNames.clear();
  fileNames.emplace_back(argv[9]);
  readerRef->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality<ImageType>(reader->GetOutput(), readerRef->GetOutput(), 1e-8, 100, 2.0);

  ///////////////////// Hnc file format
  fileNames.clear();
  fileNames.emplace_back(argv[5]);

  // Varian geometry
  geoTargReader->SetProjectionsFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoTargReader->UpdateOutputData());

  // Reference geometry
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename(argv[10]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoRefReader->GenerateOutputInformation())

  // 1. Check geometries
  CheckGeometries(geoTargReader->GetGeometry(), geoRefReader->GetOutputObject());

  // ******* COMPARING projections *******

  // Varian projections reader
  reader->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->UpdateLargestPossibleRegion());

  // Reference projections reader
  fileNames.clear();
  fileNames.emplace_back(argv[11]);
  readerRef->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->UpdateLargestPossibleRegion());

  // 2. Compare read projections
  CheckImageQuality<ImageType>(reader->GetOutput(), readerRef->GetOutput(), 1e-8, 100, 2.0);

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
