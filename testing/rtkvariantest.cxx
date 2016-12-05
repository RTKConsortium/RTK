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

int main(int, char** )
{
  std::vector<std::string> fileNames;
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Input/Varian/raw.hnd") );

  // Varian geometry
  rtk::VarianObiGeometryReader::Pointer geoTargReader;
  geoTargReader = rtk::VarianObiGeometryReader::New();
  geoTargReader->SetXMLFileName( std::string(RTK_DATA_ROOT) +
                                 std::string("/Input/Varian/acqui.xml") );
  geoTargReader->SetProjectionsFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoTargReader->UpdateOutputData() );

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename( std::string(RTK_DATA_ROOT) +
                             std::string("/Baseline/Varian/geometry.xml") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoRefReader->GenerateOutputInformation() )

  // 1. Check geometries
  CheckGeometries(geoTargReader->GetGeometry(), geoRefReader->GetOutputObject() );

  // ******* COMPARING projections *******
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > ImageType;

  // Varian projections reader
  typedef rtk::ProjectionsReader< ImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() );

  // Reference projections reader
  ReaderType::Pointer readerRef = ReaderType::New();
  fileNames.clear();
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Baseline/Varian/attenuation.mha") );
  readerRef->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality< ImageType >(reader->GetOutput(), readerRef->GetOutput(), 1e-8, 100, 2.0);

  ///////////////////// Xim file format
  fileNames.clear();
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Input/Varian/Proj_00000.xim") );

  // Varian geometry
  rtk::VarianProBeamGeometryReader::Pointer geoProBeamReader;
  geoProBeamReader = rtk::VarianProBeamGeometryReader::New();
  geoProBeamReader->SetXMLFileName( std::string(RTK_DATA_ROOT) +
                                    std::string("/Input/Varian/acqui_probeam.xml") );
  geoProBeamReader->SetProjectionsFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoProBeamReader->UpdateOutputData() );

  // Reference geometry
  geoRefReader->SetFilename( std::string(RTK_DATA_ROOT) +
                             std::string("/Baseline/Varian/geometryProBeam.xml") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoRefReader->GenerateOutputInformation() )

  // 1. Check geometries
  CheckGeometries(geoProBeamReader->GetGeometry(), geoRefReader->GetOutputObject() );

  // ******* COMPARING projections *******
  // Varian projections reader
  reader->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() );

  // Reference projections reader
  fileNames.clear();
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Baseline/Varian/attenuationProBeam.mha") );
  readerRef->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality< ImageType >(reader->GetOutput(), readerRef->GetOutput(), 1e-8, 100, 2.0);

  // If both succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
