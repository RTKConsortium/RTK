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

int
main(int argc, char * argv[])
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  projection.edf  reference.mha" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 3;
  using ImageType = itk::Image<float, Dimension>;

  // 1. ESRF / Edf projections reader
  using ReaderType = rtk::ProjectionsReader<ImageType>;
  auto                     reader = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.emplace_back(argv[1]);
  reader->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update());

  // Reference projections reader
  auto readerRef = ReaderType::New();
  fileNames.clear();
  fileNames.emplace_back(argv[2]);
  readerRef->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality<ImageType>(reader->GetOutput(), readerRef->GetOutput(), 0.00000001, 100, 2.0);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
