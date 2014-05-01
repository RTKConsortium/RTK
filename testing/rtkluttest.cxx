#include "rtkTestConfiguration.h"
#include "rtkProjectionsReader.h"
#include "rtkMacro.h"
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkCastImageFilter.h>

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
  if (ErrorPerPixel > 1.6e-7)
    {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 1.6e-7" << std::endl;
    exit( EXIT_FAILURE);
    }
  if (PSNR < 100.)
    {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 100" << std::endl;
    exit( EXIT_FAILURE);
    }
}
#endif

/**
 * \file rtkluttest.cxx
 *
 * \brief Test rtk::LookupTableImageFilter
 *
 * This test float and double lookup table filters. It compares the output of
 * the Elekta lookup table with the values of the same lut casted to float and
 * double.
 *
 * \author Simon Rit
 */

int main(int, char** )
{

  // Elekta projections reader
  typedef itk::Image< unsigned short, 3 > ShortImageType;
  typedef rtk::ProjectionsReader< ShortImageType > ReaderType;
  ReaderType::Pointer r = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Input/Elekta/raw.his") );
  r->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( r->Update() );

  /***** Float *****/
  typedef itk::Image <float, 3> FloatImageType;
  typedef rtk::ElektaSynergyLookupTableImageFilter<FloatImageType> ShortFloatLUTType;
  ShortFloatLUTType::Pointer sflut = ShortFloatLUTType::New();
  sflut->SetInput( r->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( sflut->Update() );

  typedef itk::CastImageFilter<ShortImageType, FloatImageType> FloatCastType;
  FloatCastType::Pointer fCast = FloatCastType::New();
  fCast->SetInput( r->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fCast->Update() );

  typedef itk::Image <float, 1> FloatLUTType;
  typedef itk::CastImageFilter<ShortFloatLUTType::LookupTableType, FloatLUTType> FloatLUTCastType;
  FloatLUTCastType::Pointer flCast = FloatLUTCastType::New();
  flCast->SetInput( sflut->GetLookupTable() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( flCast->Update() );

  typedef rtk::LookupTableImageFilter<FloatImageType, FloatImageType> FloatLUTTypea;
  FloatLUTTypea::Pointer flut = FloatLUTTypea::New();
  flut->SetInput( fCast->GetOutput() );
  flut->SetLookupTable( flCast->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( flut->Update() );

  CheckImageQuality< FloatImageType >(flut->GetOutput(), sflut->GetOutput());

  /***** Double *****/
  typedef itk::Image <float, 3> DoubleImageType;
  typedef rtk::ElektaSynergyLookupTableImageFilter<DoubleImageType> ShortDoubleLUTType;
  ShortDoubleLUTType::Pointer sdlut = ShortDoubleLUTType::New();
  sdlut->SetInput( r->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( sdlut->Update() );

  typedef itk::CastImageFilter<ShortImageType, DoubleImageType> DoubleCastType;
  DoubleCastType::Pointer dCast = DoubleCastType::New();
  dCast->SetInput( r->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dCast->Update() );

  typedef itk::Image <float, 1> DoubleLUTType;
  typedef itk::CastImageFilter<ShortDoubleLUTType::LookupTableType, DoubleLUTType> DoubleLUTCastType;
  DoubleLUTCastType::Pointer dlCast = DoubleLUTCastType::New();
  dlCast->SetInput( sdlut->GetLookupTable() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dlCast->Update() );

  typedef rtk::LookupTableImageFilter<DoubleImageType, DoubleImageType> DoubleLUTTypea;
  DoubleLUTTypea::Pointer dlut = DoubleLUTTypea::New();
  dlut->SetInput( dCast->GetOutput() );
  dlut->SetLookupTable( dlCast->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dlut->Update() );

  CheckImageQuality< DoubleImageType >(dlut->GetOutput(), sdlut->GetOutput());

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
