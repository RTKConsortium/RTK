#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkOraGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

#include <itkRegularExpressionSeriesFileNames.h>

/**
 * \file rtkoratest.cxx
 *
 * \brief Functional tests for classes managing Ora data (radART / medPhoton)
 *
 * This test reads and verifies the geometry from an ora projection.
 *
 * \author Simon Rit
 */

int main(int, char** )
{
  std::cout << "Testing geometry..." << std::endl;

  // Ora geometry
  std::vector<std::string> filenames;
  filenames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Input/Ora/0_afterLog.ora.xml") );
  rtk::OraGeometryReader::Pointer geoTargReader;
  geoTargReader = rtk::OraGeometryReader::New();
  geoTargReader->SetProjectionsFileNames( filenames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoTargReader->UpdateOutputData() );

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename( std::string(RTK_DATA_ROOT) +
                             std::string("/Baseline/Ora/geometry.xml") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoRefReader->GenerateOutputInformation() )

  // Check geometries
  CheckGeometries(geoTargReader->GetGeometry(), geoRefReader->GetOutputObject() );

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
