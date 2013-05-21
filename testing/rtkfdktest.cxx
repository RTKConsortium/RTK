#include <itkImageRegionConstIterator.h>
#include <itkStreamingImageFilter.h>

#include "rtkTestConfiguration.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"

#ifdef USE_CUDA
#  include "rtkCudaFDKConeBeamReconstructionFilter.h"
#elif USE_OPENCL
#  include "rtkOpenCLFDKConeBeamReconstructionFilter.h"
#else
#  include "rtkFDKConeBeamReconstructionFilter.h"
#endif

template<class TImage>
#if FAST_TESTS_NO_CHECKS
void CheckImageQuality(typename TImage::Pointer itkNotUsed(recon), typename TImage::Pointer itkNotUsed(ref))
{
}
#endif
#if !(FAST_TESTS_NO_CHECKS)
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
  if (ErrorPerPixel > 0.03)
  {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 0.03." << std::endl;
    exit( EXIT_FAILURE);
  }
  if (PSNR < 26.)
  {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 26" << std::endl;
    exit( EXIT_FAILURE);
  }
}
#endif

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfProjectionImages = 3;
#else
  const unsigned int NumberOfProjectionImages = 180;
#endif

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 32;
  size[1] = 32;
  size[2] = 32;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
#else
  size[0] = 128;
  size[1] = 128;
  size[2] = 128;
  spacing[0] = 2.;
  spacing[1] = 2.;
  spacing[2] = 2.;
#endif
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -254.;
  origin[1] = -254.;
  origin[2] = -254.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 32;
  size[1] = 32;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 32.;
  spacing[1] = 32.;
  spacing[2] = 32.;
#else
  size[0] = 128;
  size[1] = 128;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
#endif
  projectionsSource->SetOrigin( origin );
  projectionsSource->SetSpacing( spacing );
  projectionsSource->SetSize( size );
  projectionsSource->SetConstant( 0. );

  std::cout << "\n\n****** Case 1: No streaming ******" << std::endl;

  // Geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

  // Shepp Logan projections filter
  typedef rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  SLPType::Pointer slp=SLPType::New();
  slp->SetInput( projectionsSource->GetOutput() );
  slp->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( slp->Update() );

  // Create a reference object (in this case a 3D phantom reference).
  typedef rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType> DSLType;
  DSLType::Pointer dsl = DSLType::New();
  dsl->SetInput( tomographySource->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() )

  // FDK reconstruction filtering
#ifdef USE_CUDA
  typedef rtk::CudaFDKConeBeamReconstructionFilter                FDKType;
#elif USE_OPENCL
  typedef rtk::OpenCLFDKConeBeamReconstructionFilter              FDKType;
#else
  typedef rtk::FDKConeBeamReconstructionFilter< OutputImageType > FDKType;
#endif
  FDKType::Pointer feldkamp = FDKType::New();
  feldkamp->SetInput( 0, tomographySource->GetOutput() );
  feldkamp->SetInput( 1, slp->GetOutput() );
  feldkamp->SetGeometry( geometry );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );


  // FOV
  typedef rtk::FieldOfViewImageFilter<OutputImageType, OutputImageType> FOVFilterType;
  FOVFilterType::Pointer fov=FOVFilterType::New();
  fov->SetInput(0, feldkamp->GetOutput());
  fov->SetProjectionsStack(slp->GetOutput());
  fov->SetGeometry( geometry );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fov->Update() );

  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput());
  std::cout << "Test PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: perpendicular direction ******" << std::endl;

  ConstantImageSourceType::OutputImageType::DirectionType direction;
  direction[0][0] = 0;
  direction[0][1] = 1;
  direction[0][2] = 0;
  direction[1][0] = -1;
  direction[1][1] = 0;
  direction[1][2] = 0;
  direction[2][0] = 0;
  direction[2][1] = 0;
  direction[2][2] = 1;
  tomographySource->SetDirection(direction);
  origin[0] = -127.;
  origin[1] =  127.;
  origin[2] = -127.;
  tomographySource->SetOrigin( origin );
  fov->GetOutput()->ResetPipeline();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fov->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() )

  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput());
  std::cout << "Test PASSED! " << std::endl;

   std::cout << "\n\n****** Case 3: 45 degree tilt direction ******" << std::endl;

  direction[0][0] = 0.70710678118;
  direction[0][1] = -0.70710678118;
  direction[0][2] = 0.70710678118;
  direction[1][0] = 0.70710678118;
  direction[1][1] = 0;
  direction[1][2] = 0;
  direction[2][0] = 0;
  direction[2][1] = 0;
  direction[2][2] = 1;
  tomographySource->SetDirection(direction);
  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;
  tomographySource->SetOrigin( origin );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() )
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fov->Update() );

  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput());
  std::cout << "Test PASSED! " << std::endl;

 std::cout << "\n\n****** Case 4: streaming ******" << std::endl;

  // Make sure that the data will be recomputed by releasing them
  fov->GetOutput()->ReleaseData();

  typedef itk::StreamingImageFilter<OutputImageType, OutputImageType> StreamingType;
  StreamingType::Pointer streamer = StreamingType::New();
  streamer->SetInput(0, fov->GetOutput());
  streamer->SetNumberOfStreamDivisions(8);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( streamer->Update() );

  CheckImageQuality<OutputImageType>(streamer->GetOutput(), dsl->GetOutput());
  std::cout << "Test PASSED! " << std::endl;

  std::cout << "\n\n****** Case 5: small ROI ******" << std::endl;
  origin[0] = -5.;
  origin[1] = -13.;
  origin[2] = -20.;
  size[0] = 64;
  size[1] = 64;
  size[2] = 64;
  tomographySource->SetOrigin( origin );
  tomographySource->SetSize( size );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fov->UpdateLargestPossibleRegion() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->UpdateLargestPossibleRegion() )
  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput());
  std::cout << "Test PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
