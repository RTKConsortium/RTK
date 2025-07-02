#include "rtkTest.h"
#include "rtkImportImageFilter.h"
#ifdef USE_CUDA
#  include "itkCudaImage.h"
#endif

/**
 * \file rtkimporttest.cxx
 *
 * \brief Functional test for the classes performing an import
 *
 * This test perfoms an import of several vector types.
 * Compares the obtained result with the reference raw vector.
 *
 * \author Marc Vila
 */

template <class TImage>
#if FAST_TESTS_NO_CHECKS
void
CheckError(typename TImage::Pointer     itkNotUsed(recon),
           typename TImage::PixelType * itkNotUsed(ref),
           double                       itkNotUsed(ErrorPerPixelTolerance),
           double                       itkNotUsed(PSNRTolerance),
           double                       itkNotUsed(RefValueForPSNR))
{}
#else
void
CheckError(typename TImage::Pointer     recon,
           typename TImage::PixelType * ref,
           double                       ErrorPerPixelTolerance,
           double                       PSNRTolerance,
           double                       RefValueForPSNR)
{
  using ImageIteratorType = itk::ImageRegionConstIterator<TImage>;
  ImageIteratorType itTest(recon, recon->GetBufferedRegion());

  using ErrorType = double;
  ErrorType TestError = 0.;
  ErrorType EnerError = 0.;

  unsigned int k = 0;
  itTest.GoToBegin();
  while (!itTest.IsAtEnd())
  {
    typename TImage::PixelType TestVal = itTest.Get();
    TestError += itk::Math::abs(ErrorType(ref[k] - TestVal));
    EnerError += std::pow(ErrorType(ref[k] - TestVal), 2.);
    ++itTest;
    ++k;
  }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError / recon->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError / recon->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "MSE = " << MSE << std::endl;
  // PSNR
  ErrorType PSNR = 20 * log10(RefValueForPSNR) - 10 * log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (RefValueForPSNR - ErrorPerPixel) / RefValueForPSNR;
  std::cout << "QI = " << QI << std::endl;

  // Checking results. As a comparison with NaN always returns false,
  // this design allows to detect NaN results and cause test failure
  if (!(ErrorPerPixel < ErrorPerPixelTolerance))
  {
    std::cerr << "Test Failed, Error per pixel not valid! " << ErrorPerPixel << " instead of " << ErrorPerPixelTolerance
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!(PSNR > PSNRTolerance))
  {
    std::cerr << "Test Failed, PSNR not valid! " << PSNR << " instead of " << PSNRTolerance << std::endl;
    exit(EXIT_FAILURE);
  }
}
#endif // FAST_TESTS_NO_CHECKS

int
main(int, char **)
{
  // Raw vectors
  auto * vec_uint_2d = new unsigned int[10 * 10];
  int *  vec_int_2d = new int[10 * 10];
  auto * vec_float_2d = new float[10 * 10];
  auto * vec_double_2d = new double[10 * 10];

  // Initializing values
  for (unsigned int i = 0; i < 10 * 10; i++)
  {
    vec_uint_2d[i] = i;
    vec_int_2d[i] = i;
    vec_float_2d[i] = i * 1.01f;
    vec_double_2d[i] = i * 1.01;
  }

  std::cout << "\n\n****** Case 1: unsigned short ******" << std::endl;

  // Update median filter
  // Import
  rtk::ImportImageFilter<itk::Image<unsigned int, 2>>::RegionType volRegion;

  rtk::ImportImageFilter<itk::Image<unsigned int, 2>>::RegionType::IndexType volIndex;
  volIndex.Fill(0.0);
  volRegion.SetIndex(volIndex);

  rtk::ImportImageFilter<itk::Image<unsigned int, 2>>::RegionType::SizeType volSize;
  volSize.Fill(10);
  volRegion.SetSize(volSize);

  auto vol = rtk::ImportImageFilter<itk::Image<unsigned int, 2>>::New();
  vol->SetRegion(volRegion);
  vol->SetSpacing(itk::MakeVector(1.0, 1.0));
  vol->SetImportPointer(vec_uint_2d, 10 * 10, false);
  vol->Update();

  CheckError<itk::Image<unsigned int, 2>>(vol->GetOutput(), &(vec_uint_2d[0]), 0.5, 2.0, 999.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  auto volCuda = rtk::ImportImageFilter<itk::CudaImage<unsigned int, 2>>::New();
  volCuda->SetRegion(volRegion);
  volCuda->SetSpacing(itk::MakeVector(1.0, 1.0));
  volCuda->SetImportPointer(vec_uint_2d, 10 * 10, false);
  volCuda->Update();

  CheckError<itk::CudaImage<unsigned int, 2>>(volCuda->GetOutput(), &(vec_uint_2d[0]), 0.5, 2.0, 999.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif
  delete[] vec_uint_2d;

  std::cout << "\n\n****** Case 2: int ******" << std::endl;

  // Update median filter
  // Import
  rtk::ImportImageFilter<itk::Image<int, 2>>::RegionType volIntRegion;

  rtk::ImportImageFilter<itk::Image<int, 2>>::RegionType::IndexType volIntIndex;
  volIntIndex.Fill(0.0);
  volIntRegion.SetIndex(volIntIndex);

  rtk::ImportImageFilter<itk::Image<int, 2>>::RegionType::SizeType volIntSize;
  volIntSize.Fill(10);
  volIntRegion.SetSize(volIntSize);

  auto volInt = rtk::ImportImageFilter<itk::Image<int, 2>>::New();
  volInt->SetRegion(volIntRegion);
  volInt->SetSpacing(itk::MakeVector(1.0, 1.0));
  volInt->SetImportPointer(vec_int_2d, 10 * 10, false);
  volInt->Update();

  CheckError<itk::Image<int, 2>>(volInt->GetOutput(), &(vec_int_2d[0]), 0.5, 2.0, 999.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  auto volIntCuda = rtk::ImportImageFilter<itk::CudaImage<int, 2>>::New();
  volIntCuda->SetRegion(volIntRegion);
  volIntCuda->SetSpacing(itk::MakeVector(1.0, 1.0));
  volIntCuda->SetImportPointer(vec_int_2d, 10 * 10, false);
  volIntCuda->Update();

  CheckError<itk::CudaImage<int, 2>>(volIntCuda->GetOutput(), &(vec_int_2d[0]), 0.5, 2.0, 999.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif
  delete[] vec_int_2d;

  std::cout << "\n\n****** Case 3: float ******" << std::endl;

  // Update median filter
  // Import
  rtk::ImportImageFilter<itk::Image<float, 2>>::RegionType volFloatRegion;

  rtk::ImportImageFilter<itk::Image<float, 2>>::RegionType::IndexType volFloatIndex;
  volFloatIndex.Fill(0.0);
  volFloatRegion.SetIndex(volFloatIndex);

  rtk::ImportImageFilter<itk::Image<float, 2>>::RegionType::SizeType volFloatSize;
  volFloatSize.Fill(10);
  volFloatRegion.SetSize(volFloatSize);

  auto volFloat = rtk::ImportImageFilter<itk::Image<float, 2>>::New();
  volFloat->SetRegion(volFloatRegion);
  volFloat->SetSpacing(itk::MakeVector(1.0, 1.0));
  volFloat->SetImportPointer(vec_float_2d, 10 * 10, false);
  volFloat->Update();

  CheckError<itk::Image<float, 2>>(volFloat->GetOutput(), &(vec_float_2d[0]), 0.5, 2.0, 1008.99);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  auto volFloatCuda = rtk::ImportImageFilter<itk::CudaImage<float, 2>>::New();
  volFloatCuda->SetRegion(volFloatRegion);
  volFloatCuda->SetSpacing(itk::MakeVector(1.0, 1.0));
  volFloatCuda->SetImportPointer(vec_float_2d, 10 * 10, false);
  volFloatCuda->Update();

  CheckError<itk::CudaImage<float, 2>>(volFloatCuda->GetOutput(), &(vec_float_2d[0]), 0.5, 2.0, 1008.99);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif
  delete[] vec_float_2d;

  std::cout << "\n\n****** Case 4: double ******" << std::endl;

  // Update median filter
  // Import
  rtk::ImportImageFilter<itk::Image<double, 2>>::RegionType volDoubleRegion;

  rtk::ImportImageFilter<itk::Image<double, 2>>::RegionType::IndexType volDoubleIndex;
  volDoubleIndex.Fill(0.0);
  volDoubleRegion.SetIndex(volDoubleIndex);

  rtk::ImportImageFilter<itk::Image<double, 2>>::RegionType::SizeType volDoubleSize;
  volDoubleSize.Fill(10);
  volDoubleRegion.SetSize(volDoubleSize);

  auto volDouble = rtk::ImportImageFilter<itk::Image<double, 2>>::New();
  volDouble->SetRegion(volDoubleRegion);
  volDouble->SetSpacing(itk::MakeVector(1.0, 1.0));
  volDouble->SetImportPointer(vec_double_2d, 10 * 10, false);
  volDouble->Update();

  CheckError<itk::Image<double, 2>>(volDouble->GetOutput(), &(vec_double_2d[0]), 0.5, 2.0, 1008.99);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  auto volDoubleCuda = rtk::ImportImageFilter<itk::CudaImage<double, 2>>::New();
  volDoubleCuda->SetRegion(volDoubleRegion);
  volDoubleCuda->SetSpacing(itk::MakeVector(1.0, 1.0));
  volDoubleCuda->SetImportPointer(vec_double_2d, 10 * 10, false);
  volDoubleCuda->Update();

  CheckError<itk::CudaImage<double, 2>>(volDoubleCuda->GetOutput(), &(vec_double_2d[0]), 0.5, 2.0, 1008.99);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif
  delete[] vec_double_2d;

  return EXIT_SUCCESS;
}
