#include "rtkTestConfiguration.h"
#include "itkRandomImageSource.h"
#include "rtkTotalVariationImageFilter.h"
#include "rtkTotalVariationDenoisingBPDQImageFilter.h"
#include "rtkMacro.h"

template<class TImage>
#if FAST_TESTS_NO_CHECKS
void CheckTotalVariation(typename TImage::Pointer itkNotUsed(before), typename TImage::Pointer itkNotUsed(after))
{
}
#else
void CheckTotalVariation(typename TImage::Pointer before, typename TImage::Pointer after)
{
  typedef rtk::TotalVariationImageFilter<TImage> TotalVariationFilterType;
  typename TotalVariationFilterType::Pointer tv = TotalVariationFilterType::New();

  double totalVariationBefore;
  double totalVariationAfter;

  tv->SetInput(before);
  tv->Update();
  totalVariationBefore = tv->GetTotalVariation();
  std::cout << "Total variation before denoising is " << totalVariationBefore << std::endl;

  tv->SetInput(after);
  tv->Update();
  totalVariationAfter = tv->GetTotalVariation();
  std::cout << "Total variation after denoising is " << totalVariationAfter << std::endl;

  // Checking results
  if (totalVariationBefore/2 < totalVariationAfter)
  {
    std::cerr << "Test Failed: total variation was not reduced" << std::endl;
    exit( EXIT_FAILURE);
  }
}
#endif

/**
 * \file rtktotalvariationtest.cxx
 *
 * \brief Tests whether the Total Variation denoising filters indeed 
 * reduce the total variation of a random image
 *
 * This test generates a random volume and performs TV denoising on this 
 * volume. It measures its total variation before and after denoising and
 * compares. Note that the TV denoising filters do not minimize TV alone, 
 * but TV + a data attachment term (they compute the proximal operator of TV).
 * Nevertheless, in most cases, it is expected that the output has 
 * a lower TV than the input.
 *
 * \author Cyril Mory
 */

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;

#ifdef USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
#endif

  // Random image sources
  typedef itk::RandomImageSource< OutputImageType > RandomImageSourceType;
  RandomImageSourceType::Pointer randomVolumeSource  = RandomImageSourceType::New();

  // Image meta data
  RandomImageSourceType::PointType origin;
  RandomImageSourceType::SizeType size;
  RandomImageSourceType::SpacingType spacing;

  // Volume metadata
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
  randomVolumeSource->SetOrigin( origin );
  randomVolumeSource->SetSpacing( spacing );
  randomVolumeSource->SetSize( size );
  randomVolumeSource->SetMin( 0. );
  randomVolumeSource->SetMax( 1. );
  randomVolumeSource->SetNumberOfThreads(2); //With 1, it's deterministic

  // Update the source
  TRY_AND_EXIT_ON_ITK_EXCEPTION( randomVolumeSource->Update() );

  // Create and set the TV denoising filter
  typedef rtk::TotalVariationDenoisingBPDQImageFilter<OutputImageType> TVDenoisingFilterType;
  TVDenoisingFilterType::Pointer TVdenoising = TVDenoisingFilterType::New();
  TVdenoising->SetInput(randomVolumeSource->GetOutput());
  TVdenoising->SetNumberOfIterations(15);
  TVdenoising->SetLambda(0.005);
  bool* dimsProcessed = new bool[Dimension];
  for (int i=0; i<Dimension; i++)
  {
      dimsProcessed[i] = true;
  }
//  dimsProcessed[0]=false;
  TVdenoising->SetDimensionsProcessed(dimsProcessed);

  // Update the TV denoising filter
  TRY_AND_EXIT_ON_ITK_EXCEPTION( TVdenoising->Update() );
  
  CheckTotalVariation<OutputImageType>(randomVolumeSource->GetOutput(), TVdenoising->GetOutput());

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
