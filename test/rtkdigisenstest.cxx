#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkDigisensGeometryReader.h"
#include "rtkProjectionsReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

#include <itkRegularExpressionSeriesFileNames.h>


/**
 * \file rtkdigisenstest.cxx
 *
 * \brief Functional tests for classes managing Digisens data
 *
 * This test reads a projection and the geometry of an acquisition from a
 * Digisens acquisition and compares it to the expected results, which are
 * read from a baseline image in the MetaIO file format and a geometry file in
 * the RTK format, respectively.
 *
 * \author Simon Rit
 */

int
main(int argc, char * argv[])
{
  if (argc < 5)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  calibration.cal  projection.tif  geometry.xml  reference.mha" << std::endl;
    return EXIT_FAILURE;
  }

  // Elekta geometry
  rtk::DigisensGeometryReader::Pointer geoTargReader;
  geoTargReader = rtk::DigisensGeometryReader::New();
  geoTargReader->SetXMLFileName(argv[1]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoTargReader->UpdateOutputData());

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename(argv[3]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoRefReader->GenerateOutputInformation())

  // 1. Check geometries
  CheckGeometries(geoTargReader->GetGeometry(), geoRefReader->GetOutputObject());

  // ******* COMPARING projections *******
  constexpr unsigned int Dimension = 3;
  using ImageType = itk::Image<float, Dimension>;

  // Tif projections reader
  using ReaderType = rtk::ProjectionsReader<ImageType>;
  auto                     reader = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.emplace_back(argv[2]);
  reader->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update());

  // Reference projections reader
  auto                     readerRef = ReaderType::New();
  std::vector<std::string> fileNamesRef;
  fileNamesRef.emplace_back(argv[4]);
  readerRef->SetFileNames(fileNamesRef);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality<ImageType>(reader->GetOutput(), readerRef->GetOutput(), 2.31e-7, 100, 2.0);

  // If both succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
