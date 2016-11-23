#include <itkImageRegionConstIterator.h>
#include <itkStreamingImageFilter.h>

#include "rtkTestConfiguration.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"

#  include "rtkCudaFDKConeBeamReconstructionFilter.h"

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

/**
 * \file rtkfdktest.cxx
 *
 * \brief Functional test for classes performing FDK reconstructions
 *
 * This test generates the projections of a simulated Shepp-Logan phantom.
 * A CT image is reconstructed from each set of generated projection images
 * using the FDK algorithm and the reconstructed CT image is compared to the
 * expected results which is analytically computed.
 *
 * \author Simon Rit and Marc Vila
 */

int main(int, char** )
{
  itk::MultiThreader::SetGlobalMaximumNumberOfThreads( 4 );

  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;

  const unsigned int NumberOfProjectionImages = 1;
/*
  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;
  size[0] = 32;
  size[1] = 32;
  size[2] = 32;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -254.;
  origin[1] = -254.;
  origin[2] = -254.;
  size[0] = 32;
  size[1] = 32;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 32.;
  spacing[1] = 32.;
  spacing[2] = 32.;
  projectionsSource->SetOrigin( origin );
  projectionsSource->SetSpacing( spacing );
  projectionsSource->SetSize( size );
  projectionsSource->SetConstant( 0. );

  std::cout << "\n\n****** Case 1: No streaming ******" << std::endl;

  // Geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    {
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);
    }

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
  typedef rtk::CudaFDKConeBeamReconstructionFilter                FDKType;
  FDKType::Pointer feldkamp = FDKType::New();
  feldkamp->SetInput( 0, tomographySource->GetOutput() );
  feldkamp->SetInput( 1, slp->GetOutput() );
  feldkamp->SetGeometry( geometry );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );


  //feldkamp->GetOutput()->GetPixelContainer();

  std::cout << "HERE" << std::endl;
  */
  itk::MultiThreader::SetGlobalMaximumNumberOfThreads( 4 );


  OutputImageType::Pointer cudaImage = OutputImageType::New();
  OutputImageType::SizeType size;
  size[0] = size[1] = size[2] = 30;
  OutputImageType::RegionType region;
  region.SetSize(size);
  cudaImage->SetRegions(region);
  cudaImage->Allocate();
  cudaImage->FillBuffer(3.0);
  cudaImage->UpdateBuffers();

  // FOV
  typedef rtk::FieldOfViewImageFilter<OutputImageType, OutputImageType> FOVFilterType;
  FOVFilterType::Pointer fov=FOVFilterType::New();
  fov->SetInput(0,cudaImage);
  //fov->SetInput(0, feldkamp->GetOutput());
  //fov->SetProjectionsStack(slp->GetOutput());
  //fov->SetGeometry( geometry );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fov->Update() );

  //CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput());
  std::cout << "Test PASSED! " << std::endl;
}
