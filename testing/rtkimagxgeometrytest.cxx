#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkImagXGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkProjectionsReader.h"

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
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Generate projections names
  std::vector<std::string> FileNames;
  FileNames.push_back(std::string(RTK_DATA_ROOT) + std::string("/Input/ImagX/1.dcm"));

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

  // 2. Check projections
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( FileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() );

  // Reference projections reader
  ReaderType::Pointer readerRef = ReaderType::New();
  FileNames.clear();
  FileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Baseline/ImagX/attenuationDCM.mha") );
  readerRef->SetFileNames( FileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality< OutputImageType >(reader->GetOutput(),
                                       readerRef->GetOutput(),
                                       1e-8, 100, 2.0);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
