
#ifdef RAMP_FILTER_TEST_WITHOUT_FFTW
#  include "rtkConfiguration.h"
#  include <itkImageToImageFilter.h>
#  if defined(USE_FFTWF)
#    undef USE_FFTWF
#  endif
#  if defined(USE_FFTWD)
#    undef USE_FFTWD
#  endif
#  include "rtkFFTRampImageFilter.h"
#endif

#include <itkImageRegionConstIterator.h>

#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawQuadricImageFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkAdditiveGaussianNoiseImageFilter.h"

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
    if( TestVal != RefVal && (RefVal>=1.05 && RefVal<=1.06) )
      {
        TestError += abs(RefVal - TestVal);
        EnerError += vcl_pow(ErrorType(RefVal - TestVal), 2.);
      }
    ++itTest;
    ++itRef;
    }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError/ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError/ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "MSE = " << MSE << std::endl;
  // PSNR
  ErrorType PSNR = 20*log10(2.0) - 10*log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (2.0-ErrorPerPixel)/2.0;
  std::cout << "QI = " << QI << std::endl;

  // Checking results
  if (ErrorPerPixel > 0.0005)
  {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 0.005." << std::endl;
    exit( EXIT_FAILURE);
  }
  if (PSNR < 23.)
  {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 23." << std::endl;
    exit( EXIT_FAILURE);
  }
}

int main(int argc, char* argv[])
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  const unsigned int NumberOfProjectionImages = 360;

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  origin[0] = -127.5;
  origin[1] = -127.5;
  origin[2] = -127.5;
  size[0] = 256;
  size[1] = 256;
  size[2] = 256;
  spacing[0] = 1.;
  spacing[1] = 1.;
  spacing[2] = 1.;
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -255;
  origin[1] = -255;
  origin[2] = -255;
  size[0] = 256;
  size[1] = 256;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 2.;
  spacing[1] = 2.;
  spacing[2] = 2.;
  projectionsSource->SetOrigin( origin );
  projectionsSource->SetSpacing( spacing );
  projectionsSource->SetSize( size );
  projectionsSource->SetConstant( 0. );

  // Geometry
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj);

  // Shepp Logan projections filter
  typedef rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  SLPType::Pointer slp=SLPType::New();
  slp->SetInput( projectionsSource->GetOutput() );
  slp->SetGeometry(geometry);
  slp->InPlaceOff();

  // Add noise
  typedef rtk::AdditiveGaussianNoiseImageFilter< OutputImageType > NIFType;
  NIFType::Pointer noisy=NIFType::New();
  noisy->SetInput( slp->GetOutput() );
  noisy->SetMean( 0.0 );
  noisy->SetStandardDeviation( 0.5 );

  // Create a reference object (in this case a 3D phantom reference).
  typedef rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType> DSLType;
  DSLType::Pointer dsl = DSLType::New();
  dsl->SetInput( tomographySource->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() );

  // FDK reconstruction filtering
  typedef rtk::FDKConeBeamReconstructionFilter< OutputImageType > FDKCPUType;
  FDKCPUType::Pointer feldkamp = FDKCPUType::New();
  feldkamp->SetInput( 0, tomographySource->GetOutput() );
  feldkamp->SetInput( 1, noisy->GetOutput() );
  feldkamp->SetGeometry( geometry );
  feldkamp->GetRampFilter()->SetHannCutFrequency(0.8);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );

  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  exit(EXIT_SUCCESS);
}
