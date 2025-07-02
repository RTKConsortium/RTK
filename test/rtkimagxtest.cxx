#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkImagXGeometryReader.h"
#include "rtkProjectionsReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

/**
 * \file rtkimagxtest.cxx
 *
 * \brief Functional tests for classes managing iMagX data
 *
 * This test reads projections and the geometry of an acquisition from an
 * IBA CBCT acquisition and compares it to the expected results, which are
 * read from a baseline image in the MetaIO file format.
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
              << "  projection.dcm calibration.xml room.xml imagX.xml geometry.xml reference.mha DCMreference.mha"
              << std::endl;
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 3;
  using ImageType = itk::Image<float, Dimension>;

  // Generate projections names
  std::vector<std::string> FileNames;
  FileNames.emplace_back(argv[1]);

  // Create geometry reader
  auto imagxReader = rtk::ImagXGeometryReader<ImageType>::New();
  imagxReader->SetProjectionsFileNames(FileNames);
  imagxReader->SetCalibrationXMLFileName(argv[2]);
  imagxReader->SetRoomXMLFileName(argv[3]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(imagxReader->UpdateOutputData());

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename(argv[5]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geoRefReader->GenerateOutputInformation())

  // Check geometries
  CheckGeometries(imagxReader->GetGeometry(), geoRefReader->GetOutputObject());

  std::cout << "Checking one projection in xml format..." << std::endl;

  // ImagX projections reader
  using ReaderType = rtk::ProjectionsReader<ImageType>;
  ReaderType::ShrinkFactorsType binning;
  binning.Fill(2);
  binning[2] = 1;
  ReaderType::OutputImageSizeType crop;
  crop.Fill(4);
  crop[2] = 0;
  auto                     reader = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.emplace_back(argv[4]);
  reader->SetFileNames(fileNames);
  reader->SetShrinkFactors(binning);
  reader->SetLowerBoundaryCropSize(crop);
  reader->SetUpperBoundaryCropSize(crop);
  reader->SetNonNegativityConstraintThreshold(20);
  reader->SetI0(0);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update());

  // Reference projections reader
  auto readerRef = ReaderType::New();
  fileNames.clear();
  fileNames.emplace_back(argv[6]);
  readerRef->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality<ImageType>(reader->GetOutput(), readerRef->GetOutput(), 1e-8, 100, 2.0);

  std::cout << std::endl << std::endl << "Checking one projection in dcm format..." << std::endl;
  fileNames.clear();
  fileNames.emplace_back(argv[1]);
  reader->SetFileNames(fileNames);
  reader->SetShrinkFactors(binning);
  reader->SetLowerBoundaryCropSize(crop);
  reader->SetUpperBoundaryCropSize(crop);
  reader->SetNonNegativityConstraintThreshold(20);
  reader->SetI0(0);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update());

  // Reference projections reader
  fileNames.clear();
  fileNames.emplace_back(argv[7]);
  readerRef->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality<ImageType>(reader->GetOutput(), readerRef->GetOutput(), 1e-8, 100, 2.0);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
