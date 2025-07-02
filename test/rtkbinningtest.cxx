#include "rtkTest.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "rtkConstantImageSource.h"
#include <itkBinShrinkImageFilter.h>

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

int
main(int, char **)
{
  constexpr unsigned int Dimension = 2;
  using OutputImageType = itk::Image<unsigned short, Dimension>;

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;

  // Create constant image of value 2 and reference image.
  auto imgIn = ConstantImageSourceType::New();
  auto imgRef = ConstantImageSourceType::New();

  imgIn->SetOrigin(itk::MakePoint(-7, -7));
  imgIn->SetSpacing(itk::MakeVector(1., 1.));
  imgIn->SetSize(itk::MakeSize(8, 8));
  imgIn->SetConstant(2);
  imgIn->UpdateLargestPossibleRegion();

  // Adapting reference for case 1
  imgRef->SetOrigin(itk::MakePoint(-7, -7));
  imgRef->SetSpacing(itk::MakeVector(2., 2.));
  imgRef->SetSize(itk::MakeSize(4, 4));
  imgRef->SetConstant(2);
  imgRef->UpdateLargestPossibleRegion();

  // Binning filter
  auto bin = itk::BinShrinkImageFilter<OutputImageType, OutputImageType>::New();

  std::cout << "\n\n****** Case 1: binning 2x2 ******" << std::endl;

  // Update binning filter
  bin->SetInput(imgIn->GetOutput());
  bin->SetShrinkFactors(itk::MakeVector(2, 2));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(bin->UpdateLargestPossibleRegion());

  CheckImageQuality<OutputImageType>(bin->GetOutput(), imgRef->GetOutput(), 0.001, 120, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: binning 1x2 ******" << std::endl;

  imgIn->UpdateLargestPossibleRegion();

  // Adapting reference for case 2
  imgRef->SetSpacing(itk::MakeVector(1., 2.));
  imgRef->SetSize(itk::MakeSize(8, 4));
  imgRef->UpdateLargestPossibleRegion();

  // Update binning filter
  bin->SetShrinkFactors(itk::MakeVector(1, 2));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(bin->UpdateLargestPossibleRegion());

  CheckImageQuality<OutputImageType>(bin->GetOutput(), imgRef->GetOutput(), 0.001, 120, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 3: binning 2x1 ******" << std::endl;

  imgIn->UpdateLargestPossibleRegion();

  // Adapting reference for case 3
  imgRef->SetSpacing(itk::MakeVector(2., 1.));
  imgRef->SetSize(itk::MakeSize(4, 8));
  imgRef->UpdateLargestPossibleRegion();

  // Update binning filter
  bin->SetInput(imgIn->GetOutput());
  bin->SetShrinkFactors(itk::MakeVector(2, 1));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(bin->UpdateLargestPossibleRegion());

  CheckImageQuality<OutputImageType>(bin->GetOutput(), imgRef->GetOutput(), 0.001, 120, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
