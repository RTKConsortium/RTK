
#include "rtkTestConfiguration.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"

#include "rtkBinningImageFilter.h"

template<class TImage>
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
  if (ErrorPerPixel > 1.28)
    {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 1.28" << std::endl;
    exit( EXIT_FAILURE);
    }
  if (PSNR < 44.)
    {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 44" << std::endl;
    exit( EXIT_FAILURE);
    }
}

/**
 * \file rtkbinningtest.cxx
 *
 * \brief Functional test for the classes performing binning
 *
 * This test perfoms a binning on a 2D image with binning factors 2x2. Compares
 * the obtained result with a reference image previously calculated.
 *
 * \author Marc Vila
 */

int main(int , char** )
{
  const unsigned int Dimension = 2;
  typedef unsigned short                           OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size, sizeRef;
  ConstantImageSourceType::SpacingType spacing, spacingRef;

  // Create constant image of value 2 and reference image.
  const ConstantImageSourceType::Pointer imgIn  = ConstantImageSourceType::New();
  const ConstantImageSourceType::Pointer imgRef = ConstantImageSourceType::New();

  origin[0] = -126;
  origin[1] = -126;
  size[0] = 4;
  size[1] = 4;
  spacing[0] = 64.;
  spacing[1] = 64.;

  imgIn->SetOrigin( origin );
  imgIn->SetSpacing( spacing );
  imgIn->SetSize( size );
  imgIn->SetConstant( 2 );
  imgIn->UpdateOutputInformation();

  sizeRef[0] = 2;
  sizeRef[1] = 2;
  spacingRef[0] = 128.;
  spacingRef[1] = 128.;

  imgRef->SetOrigin( origin );
  imgRef->SetSpacing( spacingRef );
  imgRef->SetSize( sizeRef );
  imgRef->SetConstant( 2 );
  imgRef->UpdateOutputInformation();

  // Binning filter
  typedef rtk::BinningImageFilter BINType;
  BINType::Pointer bin = BINType::New();

  std::cout << "\n\n****** Case 1: binning 2x2 ******" << std::endl;

  // Update binning filter
  itk::Vector<unsigned int,2> binning_factors;
  binning_factors[0]=2;
  binning_factors[1]=2;
  bin->SetInput( imgIn->GetOutput() );
  bin->SetBinningFactors(binning_factors);
  bin->Update();

  CheckImageQuality<OutputImageType>(bin->GetOutput(), imgRef->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

//  std::cout << "\n\n****** Case 2: binning 1x2 ******" << std::endl;

//  // Adpating reference
//  sizeRef[0] = 4;
//  sizeRef[1] = 2;
//  spacingRef[0] = 64.;
//  spacingRef[1] = 128.;

//  imgRef->SetSpacing( spacingRef );
//  imgRef->SetSize( sizeRef );
//  imgRef->SetConstant( 4 );

//  // Update binning filter
//  binning_factors[0]=1;
//  binning_factors[1]=2;
//  bin->SetInput( imgIn->GetOutput() );
//  bin->SetBinningFactors(binning_factors);
//  bin->Update();

//  CheckImageQuality<OutputImageType>(bin->GetOutput(), imgRef->GetOutput());
//  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
