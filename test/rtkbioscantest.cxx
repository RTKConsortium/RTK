#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkBioscanGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

#include <itkRegularExpressionSeriesFileNames.h>

/**
 * \file rtkbioscantest.cxx
 *
 * \brief Functional tests for classes managing Bioscan NanoSPECT/CT data
 *
 * This test reads and verifies the geometry from a bioscan projection.
 *
 * \author Simon Rit
 */

int main(int argc, char*argv[])
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " bioscan.dcm refGeometry.xml" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Testing geometry..." << std::endl;

  // Ora geometry
  std::vector<std::string> filenames;
  filenames.push_back(argv[1]);
  rtk::BioscanGeometryReader::Pointer geoTargReader;
  geoTargReader = rtk::BioscanGeometryReader::New();
  geoTargReader->SetProjectionsFileNames( filenames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoTargReader->UpdateOutputData() );

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename(argv[2]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoRefReader->GenerateOutputInformation() )

  // Check geometries
  CheckGeometries(geoTargReader->GetGeometry(), geoRefReader->GetOutputObject() );

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
