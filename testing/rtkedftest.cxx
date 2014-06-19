#include "rtkTest.h"
#include "rtkProjectionsReader.h"
#include "rtkMacro.h"

/**
 * \file rtkedftest.cxx
 *
 * \brief Functional test for classes managing ESRF/Edf data
 *
 * This test reads a projection of an acquisition of ESRF/Edf type
 * and compares it to the expected results, which are read from a baseline
 * image in the MetaIO file format.
 *
 * \author Simon Rit
 */

int main(int, char** )
{
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > ImageType;

  // 1. ESRF / Edf projections reader
  typedef rtk::ProjectionsReader< ImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Input/ESRF/raw.edf") );
  reader->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() );

  // Reference projections reader
  ReaderType::Pointer readerRef = ReaderType::New();
  fileNames.clear();
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Baseline/ESRF/attenuation.mha") );
  readerRef->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality< ImageType >(reader->GetOutput(), readerRef->GetOutput(), 0.00000001, 100, 2.0);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
