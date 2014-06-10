#include "rtkTest.h"
#include "rtkProjectionsReader.h"
#include "rtkMacro.h"
#include "rtkDigisensGeometryReader.h"

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

int main(int, char** )
{
  // Elekta geometry
  rtk::DigisensGeometryReader::Pointer geoTargReader;
  geoTargReader = rtk::DigisensGeometryReader::New();
  geoTargReader->SetXMLFileName( std::string(RTK_DATA_ROOT) +
                                 std::string("/Input/Digisens/calibration.cal") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoTargReader->UpdateOutputData() );

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename( std::string(RTK_DATA_ROOT) + 
                             std::string("/Baseline/Digisens/geometry.xml") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoRefReader->GenerateOutputInformation() )

  // 1. Check geometries
  CheckGeometries(geoTargReader->GetGeometry(), geoRefReader->GetOutputObject() );

  // ******* COMPARING projections *******
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > ImageType;

  // Tif projections reader
  typedef rtk::ProjectionsReader< ImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Input/Digisens/ima0010.tif") );
  reader->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() );

  // Reference projections reader
  ReaderType::Pointer readerRef = ReaderType::New();
  std::vector<std::string> fileNamesRef;
  fileNamesRef.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Baseline/Digisens/attenuation.mha") );
  readerRef->SetFileNames( fileNamesRef );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality< ImageType >(reader->GetOutput(), readerRef->GetOutput(), 2.31e-7, 100, 2.0);

  // If both succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
