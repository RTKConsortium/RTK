#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkOraGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkProjectionsReader.h"
#include "rtkMaskCollimationImageFilter.h"

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

int
main(int argc, char * argv[])
{
  if (argc < 5)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "oraGeometry.xml refGeometry.xml"
              << " oraGeometry_yawtilt.xml refGeometry_yawtilt.xml"
              << " oraGeometry_yaw.xml refGeometry_yaw.xml"
              << " refGeometry_optitrack.xml"
              << " reference.mha" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Testing geometry..." << std::endl;

  // Ora geometry
  std::vector<std::string> filenames;
  filenames.emplace_back(argv[1]);
  rtk::OraGeometryReader::Pointer geoTargReader;
  geoTargReader = rtk::OraGeometryReader::New();
  geoTargReader->SetProjectionsFileNames(filenames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoTargReader->UpdateOutputData());

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename(argv[2]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoRefReader->GenerateOutputInformation())

  // Check geometries
  CheckGeometries(geoTargReader->GetGeometry(), geoRefReader->GetOutputObject());

  std::cout << "Testing geometry with tilt and yaw..." << std::endl;

  // Ora geometry
  std::vector<std::string> filenames_yawtilt;
  filenames_yawtilt.emplace_back(argv[3]);
  rtk::OraGeometryReader::Pointer geoTargReader_yawtilt;
  geoTargReader_yawtilt = rtk::OraGeometryReader::New();
  geoTargReader_yawtilt->SetProjectionsFileNames(filenames_yawtilt);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoTargReader_yawtilt->UpdateOutputData());

  // Reference geometry
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename(argv[4]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoRefReader->GenerateOutputInformation())

  // Check geometries
  CheckGeometries(geoTargReader_yawtilt->GetGeometry(), geoRefReader->GetOutputObject());

  std::cout << "Testing geometry with optitrack..." << std::endl;

  // Ora geometry
  std::vector<std::string> filenames_opti;
  filenames_opti.emplace_back(argv[3]);
  rtk::OraGeometryReader::Pointer geoTargReader_opti;
  geoTargReader_opti = rtk::OraGeometryReader::New();
  geoTargReader_opti->SetProjectionsFileNames(filenames_opti);
  geoTargReader_opti->SetOptiTrackObjectID(2);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoTargReader_opti->UpdateOutputData());

  // Reference geometry
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename(argv[5]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoRefReader->GenerateOutputInformation())

  // Check geometries
  CheckGeometries(geoTargReader_opti->GetGeometry(), geoRefReader->GetOutputObject());

  // ******* COMPARING projections *******
  std::cout << "Testing attenuation conversion..." << std::endl;

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;
  using ImageType = itk::Image<OutputPixelType, Dimension>;

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<ImageType>;
  auto reader = ReaderType::New();
  reader->SetFileNames(filenames);
  ReaderType::OutputImageSpacingType spacing;
  spacing[0] = 1.;
  spacing[1] = 2.;
  spacing[2] = 1.;
  reader->SetSpacing(spacing);
  reader->SetOrigin(itk::MakePoint(0., 0., 0.));
  ReaderType::OutputImageDirectionType direction;
  direction.SetIdentity();
  reader->SetDirection(direction);

  // Create projection image filter
  using OFMType = rtk::MaskCollimationImageFilter<ImageType, ImageType>;
  auto ofm = OFMType::New();
  ofm->SetInput(reader->GetOutput());
  ofm->SetGeometry(geoTargReader->GetModifiableGeometry());

  TRY_AND_EXIT_ON_ITK_EXCEPTION(ofm->Update());

  // Reference projections reader
  auto readerRef = ReaderType::New();
  filenames.clear();
  filenames.emplace_back(argv[6]);
  readerRef->SetFileNames(filenames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality<ImageType>(ofm->GetOutput(), readerRef->GetOutput(), 1.e-10, 100000, 2000.0);

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
