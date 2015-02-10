#include "rtkTest.h"
#include "rtkProjectionsReader.h"
#include "rtkMacro.h"
#include "rtkImagXGeometryReader.h"

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

int main(int, char** )
{
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > ImageType;

  // Generate projections names
  std::vector<std::string> FileNames;
  FileNames.push_back(std::string(RTK_DATA_ROOT) + std::string("/Input/ImagX/1.dcm"));

  // Create geometry reader
  rtk::ImagXGeometryReader<ImageType>::Pointer imagxReader = rtk::ImagXGeometryReader<ImageType>::New();
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

  // Check geometries
  CheckGeometries(imagxReader->GetGeometry(), geoRefReader->GetOutputObject() );

  std::cout << "Checking one projection in xml format..." << std::endl;

  // ImagX projections reader
  typedef rtk::ProjectionsReader< ImageType > ReaderType;
  ReaderType::ShrinkFactorsType binning;
  binning.Fill(2);
  binning[2] = 1;
  ReaderType::OutputImageSizeType crop;
  crop.Fill(4);
  crop[2] = 0;
  ReaderType::Pointer reader = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Input/ImagX/raw.xml") );
  reader->SetFileNames( fileNames );
  reader->SetShrinkFactors(binning);
  reader->SetLowerBoundaryCropSize(crop);
  reader->SetUpperBoundaryCropSize(crop);
  reader->SetNonNegativityConstraintThreshold(20);
  reader->SetI0(0);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() );

  // Reference projections reader
  ReaderType::Pointer readerRef = ReaderType::New();
  fileNames.clear();
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Baseline/ImagX/attenuation.mha") );
  readerRef->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality< ImageType >(reader->GetOutput(), readerRef->GetOutput(), 1e-8, 100, 2.0);

  std::cout << std::endl << std::endl << "Checking one projection in dcm format..." << std::endl;
  fileNames.clear();
  fileNames.push_back(std::string(RTK_DATA_ROOT) + std::string("/Input/ImagX/1.dcm"));
  reader->SetFileNames( fileNames );
  reader->SetShrinkFactors(binning);
  reader->SetLowerBoundaryCropSize(crop);
  reader->SetUpperBoundaryCropSize(crop);
  reader->SetNonNegativityConstraintThreshold(20);
  reader->SetI0(0);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() );

  // Reference projections reader
  fileNames.clear();
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Baseline/ImagX/attenuationDCM.mha") );
  readerRef->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality< ImageType >(reader->GetOutput(),
                                 readerRef->GetOutput(),
                                 1e-8, 100, 2.0);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
