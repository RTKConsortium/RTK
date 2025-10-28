#include <itkImageRegionConstIterator.h>

#include "rtkTestConfiguration.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"

#ifdef USE_CUDA
#  include "itkCudaImage.h"
#endif
#include "rtkADMMWaveletsConeBeamReconstructionFilter.h"

template <class TImage>
#if FAST_TESTS_NO_CHECKS
void
CheckImageQuality(typename TImage::Pointer itkNotUsed(recon), typename TImage::Pointer itkNotUsed(ref))
{}
#else
void
CheckImageQuality(typename TImage::Pointer recon, typename TImage::Pointer ref)
{
  using ImageIteratorType = itk::ImageRegionConstIterator<TImage>;
  ImageIteratorType itTest(recon, recon->GetBufferedRegion());
  ImageIteratorType itRef(ref, ref->GetBufferedRegion());

  using ErrorType = double;
  ErrorType TestError = 0.;
  ErrorType EnerError = 0.;

  itTest.GoToBegin();
  itRef.GoToBegin();

  while (!itRef.IsAtEnd())
  {
    typename TImage::PixelType TestVal = itTest.Get();
    typename TImage::PixelType RefVal = itRef.Get();
    TestError += itk::Math::abs(RefVal - TestVal);
    EnerError += std::pow(ErrorType(RefVal - TestVal), 2.);
    ++itTest;
    ++itRef;
  }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError / ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError / ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "MSE = " << MSE << std::endl;
  // PSNR
  ErrorType PSNR = 20 * log10(2.0) - 10 * log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (2.0 - ErrorPerPixel) / 2.0;
  std::cout << "QI = " << QI << std::endl;

  // Checking results
  if (ErrorPerPixel > 0.032)
  {
    std::cerr << "Test Failed, Error per pixel not valid! " << ErrorPerPixel << " instead of 0.08" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (PSNR < 28)
  {
    std::cerr << "Test Failed, PSNR not valid! " << PSNR << " instead of 23" << std::endl;
    exit(EXIT_FAILURE);
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

int
main(int, char **)
{
  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

#ifdef USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 180;
#endif


  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;

  auto tomographySource = ConstantImageSourceType::New();
  auto origin = itk::MakePoint(-127., -127., -127.);
#if FAST_TESTS_NO_CHECKS
  auto spacing = itk::MakeVector(252., 252., 252.);
  auto size = itk::MakeSize(2, 2, 2);
#else
  auto spacing = itk::MakeVector(4., 4., 4.);
  auto size = itk::MakeSize(64, 64, 64);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  auto projectionsSource = ConstantImageSourceType::New();
  origin = itk::MakePoint(-255., -255., -255.);
#if FAST_TESTS_NO_CHECKS
  spacing = itk::MakeVector(504., 504., 504.);
  size = itk::MakeSize(2, 2, NumberOfProjectionImages);
#else
  spacing = itk::MakeVector(8., 8., 8.);
  size = itk::MakeSize(64, 64, NumberOfProjectionImages);
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  // Geometry object
  auto geometry = rtk::ThreeDCircularProjectionGeometry::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);

  // Create ellipsoid PROJECTIONS
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType>;
  REIType::Pointer rei;

  rei = REIType::New();
  rei->SetAngle(0.);
  rei->SetDensity(1.);
  rei->SetCenter(itk::MakePoint(0., 0., 0.));
  rei->SetAxis(itk::MakeVector(90., 90., 90.));

  rei->SetInput(projectionsSource->GetOutput());
  rei->SetGeometry(geometry);

  // Update
  TRY_AND_EXIT_ON_ITK_EXCEPTION(rei->Update());

  // Create REFERENCE object (3D ellipsoid).
  auto dsl = rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType>::New();
  dsl->SetInput(tomographySource->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update())

  // ADMMWavelets reconstruction filtering
  using ADMMWaveletsType = rtk::ADMMWaveletsConeBeamReconstructionFilter<OutputImageType>;
  auto admmWavelets = ADMMWaveletsType::New();
  admmWavelets->SetInput(tomographySource->GetOutput());
  admmWavelets->SetInput(1, rei->GetOutput());
  admmWavelets->SetGeometry(geometry);
  admmWavelets->SetAlpha(10);
  admmWavelets->SetBeta(1000);
  admmWavelets->SetAL_iterations(3);
  admmWavelets->SetCG_iterations(3);
  admmWavelets->SetNumberOfLevels(3);
  admmWavelets->SetOrder(3);

  // In all cases, use the Joseph forward projector
  admmWavelets->SetForwardProjectionFilter(ADMMWaveletsType::FP_JOSEPH);

  std::cout << "\n\n****** Case 1: Voxel-Based Backprojector ******" << std::endl;

  admmWavelets->SetBackProjectionFilter(ADMMWaveletsType::BP_VOXELBASED);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(admmWavelets->Update());

  CheckImageQuality<OutputImageType>(admmWavelets->GetOutput(), dsl->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Joseph Backprojector ******" << std::endl;

  admmWavelets->SetBackProjectionFilter(ADMMWaveletsType::BP_JOSEPH);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(admmWavelets->Update());

  CheckImageQuality<OutputImageType>(admmWavelets->GetOutput(), dsl->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 3: CUDA Voxel-Based Backprojector and CUDA Forward projector ******" << std::endl;

  admmWavelets->SetForwardProjectionFilter(ADMMWaveletsType::FP_CUDARAYCAST);
  admmWavelets->SetBackProjectionFilter(ADMMWaveletsType::BP_CUDAVOXELBASED);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(admmWavelets->Update());

  CheckImageQuality<OutputImageType>(admmWavelets->GetOutput(), dsl->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  return EXIT_SUCCESS;
}
