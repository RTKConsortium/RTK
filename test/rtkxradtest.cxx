#include "rtkTest.h"
#include "rtkProjectionsReader.h"
#include "rtkMacro.h"
#include "rtkXRadGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

#include <itkRegularExpressionSeriesFileNames.h>

/**
 * \file rtkxradtest.cxx
 *
 * \brief Functional tests for classes managing X-Rad data
 *
 * This test reads a projection and the geometry of an acquisition from a
 * X-Rad acquisition and compares it to the expected results, which are
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
    std::cerr << argv[0] << "  xradGeometry.header xradProj.header refGeometry.xml reference.mha" << std::endl;
    return EXIT_FAILURE;
  }

  // Elekta geometry
  rtk::XRadGeometryReader::Pointer geoTargReader;
  geoTargReader = rtk::XRadGeometryReader::New();
  geoTargReader->SetImageFileName(argv[1]);
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

  // Elekta projections reader
  using ReaderType = rtk::ProjectionsReader<ImageType>;
  auto                     reader = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.emplace_back(argv[2]);
  reader->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update());

  // Reference projections reader
  auto readerRef = ReaderType::New();
  fileNames.clear();
  fileNames.emplace_back(argv[4]);
  readerRef->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality<ImageType>(reader->GetOutput(), readerRef->GetOutput(), 1.6e-7, 100, 2.0);

  // If both succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
