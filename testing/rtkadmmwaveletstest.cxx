#include <itkImageRegionConstIterator.h>

#include "rtkTestConfiguration.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
//#include "rtkNormalizedJosephBackProjectionImageFilter.h"

#ifdef USE_CUDA
  #include "itkCudaImage.h"
#endif
#include "rtkADMMWaveletsConeBeamReconstructionFilter.h"

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
  if (ErrorPerPixel > 0.032)
  {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 0.08" << std::endl;
    exit( EXIT_FAILURE);
  }
  if (PSNR < 28)
  {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 23" << std::endl;
    exit( EXIT_FAILURE);
  }
}
#endif

/**
 * \file rtkadmmwaveletstest.cxx
 *
 * \brief Functional test for ADMMWavelets reconstruction
 *
 * This test generates the projections of an ellipsoid and reconstructs the CT
 * image using the ADMMWavelets algorithm with different backprojectors (Voxel-Based,
 * Joseph). The generated results are compared to the
 * expected results (analytical calculation).
 *
 * \author Cyril Mory
 */

int main(int, char** )
{
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

#ifdef USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
  typedef itk::CudaImage< itk::CovariantVector 
      < OutputPixelType, Dimension >, Dimension >                GradientOutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension >     OutputImageType;
  typedef itk::Image< itk::CovariantVector 
      < OutputPixelType, Dimension >, Dimension >                GradientOutputImageType;
#endif

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
  size[0] = 2;
  size[1] = 2;
  size[2] = 2;
  spacing[0] = 252.;
  spacing[1] = 252.;
  spacing[2] = 252.;
#else
  size[0] = 64;
  size[1] = 64;
  size[2] = 64;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
#endif
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -255.;
  origin[1] = -255.;
  origin[2] = -255.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 504.;
  spacing[1] = 504.;
  spacing[2] = 504.;
#else
  size[0] = 64;
  size[1] = 64;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
#endif
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
  REIType::VectorType semiprincipalaxis, center;
  semiprincipalaxis.Fill(90.);
  center.Fill(0.);
  rei->SetAngle(0.);
  rei->SetDensity(1.);
  rei->SetCenter(center);
  rei->SetAxis(semiprincipalaxis);

  rei->SetInput( projectionsSource->GetOutput() );
  rei->SetGeometry( geometry );

  //Update
  TRY_AND_EXIT_ON_ITK_EXCEPTION( rei->Update() );

  // Create REFERENCE object (3D ellipsoid).
  typedef rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType> DEType;
  DEType::Pointer dsl = DEType::New();
  dsl->SetInput( tomographySource->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() )

  // ADMMWavelets reconstruction filtering
  typedef rtk::ADMMWaveletsConeBeamReconstructionFilter
      <OutputImageType>  ADMMWaveletsType;
  ADMMWaveletsType::Pointer admmWavelets = ADMMWaveletsType::New();
  admmWavelets->SetInput( tomographySource->GetOutput() );
  admmWavelets->SetInput(1, rei->GetOutput());
  admmWavelets->SetGeometry( geometry );
  admmWavelets->SetAlpha( 10 );
  admmWavelets->SetBeta( 1000 );
  admmWavelets->SetAL_iterations( 3 );
  admmWavelets->SetCG_iterations( 3 );
  admmWavelets->SetNumberOfLevels( 3 );
  admmWavelets->SetOrder( 3 );

  // In all cases, use the Joseph forward projector
  admmWavelets->SetForwardProjectionFilter(0);

  std::cout << "\n\n****** Case 1: Voxel-Based Backprojector ******" << std::endl;

  admmWavelets->SetBackProjectionFilter( 0 );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( admmWavelets->Update() );

  CheckImageQuality<OutputImageType>(admmWavelets->GetOutput(), dsl->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Joseph Backprojector ******" << std::endl;

  admmWavelets->SetBackProjectionFilter( 1 );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( admmWavelets->Update() );

  CheckImageQuality<OutputImageType>(admmWavelets->GetOutput(), dsl->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 3: CUDA Voxel-Based Backprojector and CUDA Forward projector ******" << std::endl;

  admmWavelets->SetForwardProjectionFilter( 2 );
  admmWavelets->SetBackProjectionFilter( 2 );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( admmWavelets->Update() );

  CheckImageQuality<OutputImageType>(admmWavelets->GetOutput(), dsl->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  return EXIT_SUCCESS;
}
