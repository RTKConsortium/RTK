#include <itkImageRegionConstIterator.h>

#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkCudaBackProjectionImageFilter.h"
#include "rtkJosephBackProjectionImageFilter.h"

#ifdef USE_CUDA
#  include "rtkCudaSARTConeBeamReconstructionFilter.h"
#else
#  include "rtkSARTConeBeamReconstructionFilter.h"
#endif

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
  if (ErrorPerPixel > 0.032)
  {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 0.032" << std::endl;
    exit( EXIT_FAILURE);
  }
  if (PSNR < 28.6)
  {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 28.6" << std::endl;
    exit( EXIT_FAILURE);
  }
}

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  const unsigned int NumberOfProjectionImages      = 180;


  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;
  size[0] = 64;
  size[1] = 64;
  size[2] = 64;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -255.;
  origin[1] = -255.;
  origin[2] = -255.;
  size[0] = 64;
  size[1] = 64;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
  projectionsSource->SetOrigin( origin );
  projectionsSource->SetSpacing( spacing );
  projectionsSource->SetSize( size );
  projectionsSource->SetConstant( 0. );

  // Geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages);

  // Create ellipsoid PROJECTIONS
  typedef rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType> REIType;
  REIType::Pointer rei;

  rei = REIType::New();
  rei->SetSemiPrincipalAxisX(0.69*128);
  rei->SetSemiPrincipalAxisZ(0.92*128);
  rei->SetSemiPrincipalAxisY(0.90*128);
  rei->SetCenterX(0.);
  rei->SetCenterZ(0.);
  rei->SetCenterY(0.);
  rei->SetRotationAngle(0.);
  rei->SetMultiplicativeConstant(1.);

  rei->SetInput( projectionsSource->GetOutput() );
  rei->SetGeometry( geometry );

  //Update
  rei->Update();

  // Create REFERENCE object (3D ellipsoid).
  typedef rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType> DEType;
  DEType::Pointer dsl = DEType::New();
  dsl->SetInput( tomographySource->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() )

  // SART reconstruction filtering
#ifdef USE_CUDA
  typedef rtk::CudaSARTConeBeamReconstructionFilter                SARTType;
#else
  typedef rtk::SARTConeBeamReconstructionFilter< OutputImageType > SARTType;
#endif
  SARTType::Pointer sart = SARTType::New();
  sart->SetInput( tomographySource->GetOutput() );
  sart->SetInput(1, rei->GetOutput());
  sart->SetGeometry( geometry );
  sart->SetNumberOfIterations( 1 );
  sart->SetLambda( 0.5 );

  std::cout << "\n\n****** Case 1: Voxel-Based Backprojector ******" << std::endl;

  rtk::BackProjectionImageFilter<OutputImageType, OutputImageType>::Pointer bp;
  bp = rtk::BackProjectionImageFilter<OutputImageType, OutputImageType>::New();
  sart->SetBackProjectionFilter( bp );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( sart->Update() );

  CheckImageQuality<OutputImageType>(sart->GetOutput(), dsl->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Joseph Backprojector ******" << std::endl;

  bp = rtk::JosephBackProjectionImageFilter<OutputImageType, OutputImageType>::New();
  sart->SetBackProjectionFilter( bp );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( sart->Update() );

  CheckImageQuality<OutputImageType>(sart->GetOutput(), dsl->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 3: CUDA Voxel-Based Backprojector ******" << std::endl;

  bp = rtk::CudaBackProjectionImageFilter::New();
  sart->SetBackProjectionFilter( bp );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( sart->Update() );

  CheckImageQuality<OutputImageType>(sart->GetOutput(), dsl->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  return EXIT_SUCCESS;
}
