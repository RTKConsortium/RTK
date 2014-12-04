#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkImagXGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

#include <itkRegularExpressionSeriesFileNames.h>

/**
 * \file rtkimagxgeometrytest.cxx
 *
 * \brief Functional tests for classes managing ImagX geometry
 *
 * This test generates an imagx geometry from an
 * IBA CBCT acquisition and compares it to the expected results, which are
 * read from a baseline geometry file in the RTK format.
 *
 * \author Marc Vila
 */

int main(int, char** )
{
  // Image Type
  typedef unsigned short OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Generate projections names
  std::vector<std::string> FileNames;
  FileNames.push_back(std::string(RTK_DATA_ROOT) + std::string("/Input/ImagX/1.dcm"));
  FileNames.push_back(std::string(RTK_DATA_ROOT) + std::string("/Input/ImagX/2.dcm"));
  FileNames.push_back(std::string(RTK_DATA_ROOT) + std::string("/Input/ImagX/3.dcm"));
  FileNames.push_back(std::string(RTK_DATA_ROOT) + std::string("/Input/ImagX/4.dcm"));

  // Create geometry reader
  rtk::ImagXGeometryReader<OutputImageType>::Pointer imagxReader = rtk::ImagXGeometryReader<OutputImageType>::New();
  imagxReader->SetProjectionsFileNames(FileNames);
  imagxReader->SetCalibrationXMLFileName(std::string(RTK_DATA_ROOT) +
                                         std::string("/Input/ImagX/calibration.xml"));
  imagxReader->SetRoomXMLFileName(std::string(RTK_DATA_ROOT) +
                                  std::string("/Input/ImagX/room.xml"));
  imagxReader->SetDetectorOffset(0);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( imagxReader->UpdateOutputData() );

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename( std::string(RTK_DATA_ROOT) + 
                             std::string("/Baseline/ImagX/geo.xml") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoRefReader->GenerateOutputInformation() )

  // 1. Check geometries
  CheckGeometries(imagxReader->GetGeometry(), geoRefReader->GetOutputObject() );

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
