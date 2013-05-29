#include "rtkTestConfiguration.h"
#include "rtkProjectionsReader.h"
#include "rtkMacro.h"

template<class TImage>
#if FAST_TESTS_NO_CHECKS
void CheckImageQuality(typename TImage::Pointer itkNotUsed(recon), typename TImage::Pointer itkNotUsed(ref))
{
}
#else
void CheckImageQuality(typename TImage::Pointer recon, typename TImage::Pointer ref)
{
  typedef itk::ImageRegionConstIterator<TImage> ImageIteratorType;
  ImageIteratorType itTest( recon, recon->GetBufferedRegion() );
  ImageIteratorType itRef( ref, ref->GetBufferedRegion() );

  typedef double ErrorType;
  ErrorType TestError = 0.;
  ErrorType EnerError = 0.;

  itTest.GoToBegin();
  itRef.GoToBegin();

  while( !itRef.IsAtEnd() )
    {
    typename TImage::PixelType TestVal = itTest.Get();
    typename TImage::PixelType RefVal = itRef.Get();

    if( TestVal != RefVal )
      {
      TestError += vcl_abs(RefVal - TestVal);
      EnerError += vcl_pow(ErrorType(RefVal - TestVal), 2.);
      }
    ++itTest;
    ++itRef;
    }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError/recon->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError/ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "MSE = " << MSE << std::endl;
  // PSNR
  ErrorType PSNR = 20*log10(255.0) - 10*log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (255.0-ErrorPerPixel)/255.0;
  std::cout << "QI = " << QI << std::endl;

  // Checking results
  if (ErrorPerPixel > 0.00000001)
    {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 1.28" << std::endl;
    exit( EXIT_FAILURE);
    }
  if (PSNR < 100.)
    {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 44" << std::endl;
    exit( EXIT_FAILURE);
    }
}
#endif

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
  CheckImageQuality< ImageType >(reader->GetOutput(), readerRef->GetOutput());

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
