#include <itkImageRegionIteratorWithIndex.h>

#include "rtkSubRegionViewImageFilter.h"
#include "rtkTest.h"

#ifdef USE_CUDA
#  include <itkCudaImage.h>
#endif

/**
 * \file rtksubregionviewimagetest.cxx
 *
 * \brief Functional test for rtk::SubRegionViewImageFilter
 *
 * This test verifies that the filter extracts a sub-region of an image by
 * sharing the input buffer (zero-copy view) when the pixels are contiguous,
 * and by copying when they are not. It also checks the size-1 dimension case,
 * which is contiguous only when it spans a singleton dimension of the input.
 *
 * \author Axel Garcia
 */

constexpr unsigned int Dimension = 3;
using PixelType = float;
#ifdef USE_CUDA
using ImageType = itk::CudaImage<PixelType, Dimension>;
#else
using ImageType = itk::Image<PixelType, Dimension>;
#endif

/* Fill an image with unique values v = i0 + i1*10 + i2*100, using absolute
 * indices (local index + indexOffset) so any region can be reproduced. */
void
FillGradient(ImageType::Pointer image, const ImageType::IndexType & indexOffset)
{
  itk::ImageRegionIteratorWithIndex<ImageType> it(image, image->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const ImageType::IndexType idx = it.GetIndex();
    it.Set(static_cast<PixelType>(idx[0] + indexOffset[0]) + static_cast<PixelType>(idx[1] + indexOffset[1]) * 10.0F +
           static_cast<PixelType>(idx[2] + indexOffset[2]) * 100.0F);
  }
}

ImageType::Pointer
CreateGradientImage(const ImageType::SizeType & size, const ImageType::IndexType & indexOffset)
{
  ImageType::Pointer image = ImageType::New();
  image->SetRegions(size);
  image->Allocate();
  FillGradient(image, indexOffset);
  return image;
}

int
rtksubregionviewimagetest(int, char *[])
{
  using ViewFilterType = rtk::SubRegionViewImageFilter<ImageType>;
  using RegionType = ImageType::RegionType;

  // Input image with known pixel values
  ImageType::Pointer input = CreateGradientImage(itk::MakeSize(4, 5, 6), itk::MakeIndex(0, 0, 0));

  // ===== Case 1: contiguous region -> zero-copy view =====
  RegionType region;
  region.SetIndex(itk::MakeIndex(0, 0, 2));
  region.SetSize(itk::MakeSize(4, 5, 3));

  auto view = ViewFilterType::New();
  view->SetInput(input);
  view->SetExtractionRegion(region);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(view->Update())

  if (!view->GetIsContiguous())
  {
    std::cerr << "Case 1 failed: region should be contiguous" << std::endl;
    return EXIT_FAILURE;
  }
  if (view->GetOutput()->GetLargestPossibleRegion() != region)
  {
    std::cerr << "Case 1 failed: wrong output region" << std::endl;
    return EXIT_FAILURE;
  }
  // The output must share the input buffer at offset 2*4*5 = 40.
  ptrdiff_t offset = view->GetOutput()->GetBufferPointer() - input->GetBufferPointer();
  if (offset != 40)
  {
    std::cerr << "Case 1 failed: expected view offset 40, got " << offset << std::endl;
    return EXIT_FAILURE;
  }
  ImageType::Pointer reference = CreateGradientImage(itk::MakeSize(4, 5, 3), itk::MakeIndex(0, 0, 2));
  CheckImageQuality<ImageType>(view->GetOutput(), reference, 0.001, 100, 432.);

  // ===== Case 2: non-contiguous region -> copy =====
  region.SetIndex(itk::MakeIndex(1, 1, 3));
  region.SetSize(itk::MakeSize(2, 2, 2));

  auto copy = ViewFilterType::New();
  copy->SetInput(input);
  copy->SetExtractionRegion(region);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(copy->Update())

  if (copy->GetIsContiguous())
  {
    std::cerr << "Case 2 failed: region should NOT be contiguous" << std::endl;
    return EXIT_FAILURE;
  }
  if (copy->GetOutput()->GetLargestPossibleRegion() != region)
  {
    std::cerr << "Case 2 failed: wrong output region" << std::endl;
    return EXIT_FAILURE;
  }
  // The copy must not point into the input buffer.
  const PixelType * inputPtr = input->GetBufferPointer();
  const PixelType * outputPtr = copy->GetOutput()->GetBufferPointer();
  if (outputPtr >= inputPtr && outputPtr < inputPtr + input->GetLargestPossibleRegion().GetNumberOfPixels())
  {
    std::cerr << "Case 2 failed: copy must not share the input buffer" << std::endl;
    return EXIT_FAILURE;
  }
  ImageType::Pointer reference2 = CreateGradientImage(itk::MakeSize(2, 2, 2), itk::MakeIndex(1, 1, 3));
  CheckImageQuality<ImageType>(copy->GetOutput(), reference2, 0.001, 100, 432.);

  // ===== Case 3: input without buffer -> metadata-only output =====
  ImageType::Pointer metaInput = ImageType::New(); // not allocated
  metaInput->SetRegions(itk::MakeSize(4, 5, 6));

  RegionType metaRegion;
  metaRegion.SetIndex(itk::MakeIndex(0, 0, 1));
  metaRegion.SetSize(itk::MakeSize(4, 5, 2));

  auto metadata = ViewFilterType::New();
  metadata->SetInput(metaInput);
  metadata->SetExtractionRegion(metaRegion);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(metadata->UpdateOutputInformation())

  if (metadata->GetOutput()->GetLargestPossibleRegion() != metaRegion)
  {
    std::cerr << "Case 3 failed: wrong output region" << std::endl;
    return EXIT_FAILURE;
  }
#ifndef USE_CUDA
  // For CudaImage, GetBufferPointer() has side effects on CudaDataManager
  // that prevent a clean nullptr check, so it is skipped for CUDA.
  if (metadata->GetOutput()->GetBufferPointer() != nullptr)
  {
    std::cerr << "Case 3 failed: expected a null output buffer" << std::endl;
    return EXIT_FAILURE;
  }
#endif

  // ===== Case 4: size-1 dimension (contiguous) -> zero-copy view =====
  ImageType::Pointer singleton = CreateGradientImage(itk::MakeSize(4, 1, 6), itk::MakeIndex(0, 0, 0));

  region.SetIndex(itk::MakeIndex(0, 0, 1));
  region.SetSize(itk::MakeSize(4, 1, 2));

  auto viewSize1 = ViewFilterType::New();
  viewSize1->SetInput(singleton);
  viewSize1->SetExtractionRegion(region);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(viewSize1->Update())

  if (!viewSize1->GetIsContiguous())
  {
    std::cerr << "Case 4 failed: region should be contiguous" << std::endl;
    return EXIT_FAILURE;
  }
  // Row-major strides for (4,1,6) are 1, 4, 4, so offset = 1*4 = 4.
  offset = viewSize1->GetOutput()->GetBufferPointer() - singleton->GetBufferPointer();
  if (offset != 4)
  {
    std::cerr << "Case 4 failed: expected view offset 4, got " << offset << std::endl;
    return EXIT_FAILURE;
  }
  ImageType::Pointer reference4 = CreateGradientImage(itk::MakeSize(4, 1, 2), itk::MakeIndex(0, 0, 1));
  CheckImageQuality<ImageType>(viewSize1->GetOutput(), reference4, 0.001, 100, 432.);

  // ===== Case 5: size-1 dimension (non-contiguous) -> copy =====
  region.SetIndex(itk::MakeIndex(2, 0, 2));
  region.SetSize(itk::MakeSize(1, 5, 3));

  auto copySize1 = ViewFilterType::New();
  copySize1->SetInput(input);
  copySize1->SetExtractionRegion(region);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(copySize1->Update())

  if (copySize1->GetIsContiguous())
  {
    std::cerr << "Case 5 failed: region should NOT be contiguous" << std::endl;
    return EXIT_FAILURE;
  }
  outputPtr = copySize1->GetOutput()->GetBufferPointer();
  if (outputPtr >= inputPtr && outputPtr < inputPtr + input->GetLargestPossibleRegion().GetNumberOfPixels())
  {
    std::cerr << "Case 5 failed: copy must not share the input buffer" << std::endl;
    return EXIT_FAILURE;
  }
  ImageType::Pointer reference5 = CreateGradientImage(itk::MakeSize(1, 5, 3), itk::MakeIndex(2, 0, 2));
  CheckImageQuality<ImageType>(copySize1->GetOutput(), reference5, 0.001, 100, 432.);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
