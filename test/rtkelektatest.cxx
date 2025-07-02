#include "rtkTest.h"
#include "rtkProjectionsReader.h"
#include "rtkMacro.h"
#include "rtkElektaSynergyGeometryReader.h"
#include "rtkElektaXVI5GeometryXMLFileReader.h"
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

int
main(int argc, char * argv[])
{
  if (argc < 8)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0]
              << "  image.DBF frame.DBF proj.his elektaGeometry.xml refGeometry.xml reference.mha refElektaGeometry.xml"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Testing geometry with FRAME.DBF..." << std::endl;

  // Elekta geometry
  rtk::ElektaSynergyGeometryReader::Pointer geoTargReader;
  geoTargReader = rtk::ElektaSynergyGeometryReader::New();
  geoTargReader->SetDicomUID("1.3.46.423632.135428.1351013645.166");
  geoTargReader->SetImageDbfFileName(argv[1]);
  geoTargReader->SetFrameDbfFileName(argv[2]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoTargReader->UpdateOutputData());

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename(argv[5]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoRefReader->GenerateOutputInformation())

  std::cout << "Testing geometry with _Frames.xml..." << std::endl;

  // Elekta geometry XVI v5
  rtk::ElektaXVI5GeometryXMLFileReader::Pointer geo5TargReader;
  geo5TargReader = rtk::ElektaXVI5GeometryXMLFileReader::New();
  geo5TargReader->SetFilename(argv[4]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geo5TargReader->GenerateOutputInformation());

  // Reference geometry v5
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geo5RefReader;
  geo5RefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geo5RefReader->SetFilename(argv[7]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geo5RefReader->GenerateOutputInformation())

  // 1. Check geometries
  CheckGeometries(geoTargReader->GetGeometry(), geoRefReader->GetOutputObject());
  CheckGeometries(geo5TargReader->GetGeometry(), geo5RefReader->GetOutputObject());

  std::cout << "Testing his file processing..." << std::endl;

  // ******* COMPARING projections *******
  constexpr unsigned int Dimension = 3;
  using ImageType = itk::Image<float, Dimension>;

  // Elekta projections reader
  using ReaderType = rtk::ProjectionsReader<ImageType>;
  auto                     reader = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.emplace_back(argv[3]);
  reader->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update());

  // Reference projections reader
  auto readerRef = ReaderType::New();
  fileNames.clear();
  fileNames.emplace_back(argv[6]);
  readerRef->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality<ImageType>(reader->GetOutput(), readerRef->GetOutput(), 1.6e-7, 100, 2.0);

  // ******* Test split of lookup table ******
  using InputImageType = itk::Image<unsigned short, 3>;
  auto r = itk::ImageFileReader<InputImageType>::New();
  r->SetFileName(argv[3]);
  r->Update();

  auto full = rtk::ElektaSynergyLookupTableImageFilter<ImageType>::New();
  full->SetInput(r->GetOutput());
  full->Update();

  auto raw = rtk::ElektaSynergyRawLookupTableImageFilter<InputImageType, InputImageType>::New();
  raw->SetInput(r->GetOutput());
  raw->Update();

  auto log = rtk::LUTbasedVariableI0RawToAttenuationImageFilter<InputImageType, ImageType>::New();
  log->SetInput(raw->GetOutput());
  log->SetI0(log->GetI0() + 1.);
  log->Update();

  // Compare the result of the full lut with the split lut
  CheckImageQuality<ImageType>(full->GetOutput(), log->GetOutput(), 1.6e-7, 100, 2.0);

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
