#include "rtkTestConfiguration.h"
#include "rtkProjectionsReader.h"
#include "rtkMacro.h"
#include "rtkTest.h"

/**
 * \file rtkimagxtest.cxx
 *
 * \brief Functional tests for classes managing iMagX data
 *
 * This test reads projections and the geometry of an acquisition from an
 * iMagX CBCT acquisition and compares it to the expected results, which are
 * read from a baseline image in the MetaIO file format.
 *
 * \author Simon Rit
 */

int main(int, char** )
{
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > ImageType;

  // ImagX projections reader
  typedef rtk::ProjectionsReader< ImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Input/ImagX/raw.xml") );
  reader->SetFileNames( fileNames );
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

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
