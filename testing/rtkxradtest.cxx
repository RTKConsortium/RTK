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

int main(int, char** )
{
  // Elekta geometry
  rtk::XRadGeometryReader::Pointer geoTargReader;
  geoTargReader = rtk::XRadGeometryReader::New();
  geoTargReader->SetImageFileName( std::string(RTK_DATA_ROOT) +
                                   std::string("/Input/XRad/SolidWater_HiGain1x1.header") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoTargReader->UpdateOutputData() );

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename( std::string(RTK_DATA_ROOT) +
                             std::string("/Baseline/XRad/geometry.xml") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoRefReader->GenerateOutputInformation() )

  // 1. Check geometries
  CheckGeometries(geoTargReader->GetGeometry(), geoRefReader->GetOutputObject() );

  // ******* COMPARING projections *******
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > ImageType;

  // Elekta projections reader
  typedef rtk::ProjectionsReader< ImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Input/XRad/SolidWater_HiGain1x1_firstProj.header") );
  reader->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() );

  // Reference projections reader
  ReaderType::Pointer readerRef = ReaderType::New();
  fileNames.clear();
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Baseline/XRad/attenuation.mha") );
  readerRef->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality< ImageType >(reader->GetOutput(), readerRef->GetOutput(), 1.6e-7, 100, 2.0);

  // If both succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
