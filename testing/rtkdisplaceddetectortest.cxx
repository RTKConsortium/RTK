#include <itkImageRegionConstIterator.h>

#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkDisplacedDetectorImageFilter.h"

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
    TestError += vcl_abs(RefVal - TestVal);
    EnerError += vcl_pow(ErrorType(RefVal - TestVal), 2.);
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
  if (ErrorPerPixel > 0.061)
  {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 0.06." << std::endl;
    exit( EXIT_FAILURE);
  }
  if (PSNR < 24.)
  {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 24." << std::endl;
    exit( EXIT_FAILURE);
  }
}

int main(int argc, char* argv[])
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  const unsigned int NumberOfProjectionImages = 180;

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;
  size[0] = 128;
  size[1] = 128;
  size[2] = 128;
  spacing[0] = 2.;
  spacing[1] = 2.;
  spacing[2] = 2.;
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -254.;
  origin[1] = -254.;
  origin[2] = -254.;
  size[0] = 128;
  size[1] = 128;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
  projectionsSource->SetOrigin( origin );
  projectionsSource->SetSpacing( spacing );
  projectionsSource->SetSize( size );
  projectionsSource->SetConstant( 0. );

  // Shepp Logan projections filter
  typedef rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  SLPType::Pointer slp=SLPType::New();
  slp->SetInput( projectionsSource->GetOutput() );

  // Displaced detector weighting
  typedef rtk::DisplacedDetectorImageFilter< OutputImageType > DDFType;
  DDFType::Pointer ddf = DDFType::New();
  ddf->SetInput( slp->GetOutput() );

  // Create a reference object (in this case a 3D phantom reference).
  typedef rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType> DSLType;
  DSLType::Pointer dsl = DSLType::New();
  dsl->SetInput( tomographySource->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() );

  // FDK reconstruction filtering
  typedef rtk::FDKConeBeamReconstructionFilter< OutputImageType > FDKCPUType;
  FDKCPUType::Pointer feldkamp = FDKCPUType::New();
  feldkamp->SetInput( 0, tomographySource->GetOutput() );
  feldkamp->SetInput( 1, ddf->GetOutput() );

  std::cout << "\n\n****** Case 1: positive offset in geometry ******" << std::endl;
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  slp->SetGeometry(geometry);
  ddf->SetGeometry( geometry );
  feldkamp->SetGeometry( geometry );
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages, 120., 0.);
  slp->Update();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput());

  std::cout << "\n\n****** Case 2: negative offset in geometry ******" << std::endl;
  geometry = GeometryType::New();
  slp->SetGeometry(geometry);
  ddf->SetGeometry( geometry );
  feldkamp->SetGeometry( geometry );
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages, -120., 0.);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput());

  std::cout << "\n\n****** Case 3: no displacement ******" << std::endl;
  geometry = GeometryType::New();
  slp->SetGeometry(geometry);
  ddf->SetGeometry( geometry );
  feldkamp->SetGeometry( geometry );
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages);
  projectionsSource->SetOrigin(origin);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput());

  std::cout << "\n\n****** Case 4: negative offset in origin ******" << std::endl;
  geometry = GeometryType::New();
  slp->SetGeometry(geometry);
  ddf->SetGeometry( geometry );
  feldkamp->SetGeometry( geometry );
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages);
  origin[0] = -400;
  projectionsSource->SetOrigin(origin);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput());

  std::cout << "\n\n****** Case 5: positive offset in origin ******" << std::endl;
  origin[0] = -100;
  projectionsSource->SetOrigin(origin);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput());

  std::cout << "\n\nTest PASSED! " << std::endl;
  exit(EXIT_SUCCESS);
}
