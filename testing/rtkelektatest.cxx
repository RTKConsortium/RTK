#include "rtkTest.h"
#include "rtkProjectionsReader.h"
#include "rtkMacro.h"
#include "rtkElektaSynergyGeometryReader.h"
#include "rtkElektaXVI5GeometryXMLFile.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

#include <itkRegularExpressionSeriesFileNames.h>

/**
 * \file rtkelektatest.cxx
 *
 * \brief Functional tests for classes managing Elekta Synergy data
 *
 * This test reads a projection and the geometry of an acquisition from an
 * Elekta Synergy acquisition and compares it to the expected results, which are
 * read from a baseline image in the MetaIO file format and a geometry file in
 * the RTK format, respectively.
 *
 * \author Simon Rit
 */

int main(int, char** )
{
  std::cout << "Testing geometry with FRAME.DBF..." << std::endl;

  // Elekta geometry
  rtk::ElektaSynergyGeometryReader::Pointer geoTargReader;
  geoTargReader = rtk::ElektaSynergyGeometryReader::New();
  geoTargReader->SetDicomUID("1.3.46.423632.135428.1351013645.166");
  geoTargReader->SetImageDbfFileName( std::string(RTK_DATA_ROOT) +
                                      std::string("/Input/Elekta/IMAGE.DBF") );
  geoTargReader->SetFrameDbfFileName( std::string(RTK_DATA_ROOT) +
                                      std::string("/Input/Elekta/FRAME.DBF") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoTargReader->UpdateOutputData() );

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename( std::string(RTK_DATA_ROOT) +
                             std::string("/Baseline/Elekta/geometry.xml") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoRefReader->GenerateOutputInformation() )

  std::cout << "Testing geometry with _Frames.xml..." << std::endl;

  // Elekta geometry XVI v5
  rtk::ElektaXVI5GeometryXMLFileReader::Pointer geo5TargReader;
  geo5TargReader = rtk::ElektaXVI5GeometryXMLFileReader::New();
  geo5TargReader->SetFilename(std::string(RTK_DATA_ROOT) +
                              std::string("/Input/Elekta/_Frames.xml"));
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geo5TargReader->GenerateOutputInformation() );

  // Reference geometry v5
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geo5RefReader;
  geo5RefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geo5RefReader->SetFilename( std::string(RTK_DATA_ROOT) +
                             std::string("/Baseline/Elekta/geometry5.xml") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geo5RefReader->GenerateOutputInformation() )

  // 1. Check geometries
  CheckGeometries(geoTargReader->GetGeometry(), geoRefReader->GetOutputObject() );
  CheckGeometries(geo5TargReader->GetGeometry(), geo5RefReader->GetOutputObject() );

  std::cout << "Testing his file processing..." << std::endl;

  // ******* COMPARING projections *******
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > ImageType;

  // Elekta projections reader
  typedef rtk::ProjectionsReader< ImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Input/Elekta/raw.his") );
  reader->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() );

  // Reference projections reader
  ReaderType::Pointer readerRef = ReaderType::New();
  fileNames.clear();
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Baseline/Elekta/attenuation.mha") );
  readerRef->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality< ImageType >(reader->GetOutput(), readerRef->GetOutput(), 1.6e-7, 100, 2.0);

  // ******* Test split of lookup table ******
  typedef unsigned short InputPixelType;
  typedef itk::Image< InputPixelType, 3 > InputImageType;
  typedef itk::ImageFileReader< InputImageType> RawReaderType;
  RawReaderType::Pointer r = RawReaderType::New();
  r->SetFileName(std::string(RTK_DATA_ROOT) +
                 std::string("/Input/Elekta/raw.his"));
  r->Update();

  typedef rtk::ElektaSynergyLookupTableImageFilter<ImageType> FullLUTType;
  FullLUTType::Pointer full = FullLUTType::New();
  full->SetInput(r->GetOutput());
  full->Update();

  typedef rtk::ElektaSynergyRawLookupTableImageFilter<InputImageType, InputImageType> RawLUTType;
  RawLUTType::Pointer raw = RawLUTType::New();
  raw->SetInput(r->GetOutput());
  raw->Update();

  typedef rtk::LUTbasedVariableI0RawToAttenuationImageFilter<InputImageType,ImageType> LogLUTType;
  LogLUTType::Pointer log = LogLUTType::New();
  log->SetInput(raw->GetOutput());
  log->SetI0(log->GetI0()+1.);
  log->Update();

  // Compare the result of the full lut with the split lut
  CheckImageQuality< ImageType >(full->GetOutput(), log->GetOutput(), 1.6e-7, 100, 2.0);

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
