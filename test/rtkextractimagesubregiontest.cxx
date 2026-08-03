#include <itkImageRegionIterator.h>

#include "rtkConstantImageSource.h"
#include "rtkExtractImageSubRegion.h"
#include "rtkTest.h"

#ifdef USE_CUDA
#  include <itkCudaImage.h>
#endif

/**
 * \file rtkextractimagesubregiontest.cxx
 *
 * \brief Functional test for rtk::ExtractImageSubRegion
 *
 * This test verifies that rtk::ExtractImageSubRegion produces a zero-copy
 * view when the requested region is contiguous along the last dimension,
 * and falls back to a copy when it is not.
 *
 * \author Axel Garcia
 */

int
rtkextractimagesubregiontest(int, char *[])
{
  constexpr unsigned int Dimension = 3;
  using PixelType = float;
#ifdef USE_CUDA
  using ImageType = itk::CudaImage<PixelType, Dimension>;
#else
  using ImageType = itk::Image<PixelType, Dimension>;
#endif
  using RegionType = itk::ImageRegion<Dimension>;
  using IndexType = itk::Index<Dimension>;
  using SizeType = itk::Size<Dimension>;

  // Create a 4x5x6 image with known pixel values
  auto source = rtk::ConstantImageSource<ImageType>::New();
  source->SetOrigin(itk::MakePoint(0., 0., 0.));
  source->SetSpacing(itk::MakeVector(1., 1., 1.));
  source->SetSize(itk::MakeSize(4, 5, 6));
  source->SetConstant(3.14f);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(source->UpdateLargestPossibleRegion());

  ImageType::Pointer input = source->GetOutput();

  // Fill with a gradient so each pixel is unique: value = x + y*10 + z*100
  itk::ImageRegionIterator<ImageType> it(input, input->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    IndexType idx = it.GetIndex();
    it.Set(static_cast<PixelType>(idx[0] + idx[1] * 10 + idx[2] * 100));
  }

  // ===== Case 1: Contiguous extraction (dims 0,1 span full input) =====
  std::cout << "\n\n****** Case 1: contiguous extraction (zero-copy) ******" << std::endl;

  RegionType contiguousRegion;
  contiguousRegion.SetIndex(itk::MakeIndex(0, 0, 2));
  contiguousRegion.SetSize(itk::MakeSize(4, 5, 3));

  ImageType::Pointer subRegion = rtk::ExtractImageSubRegion(input.GetPointer(), contiguousRegion);

  // Verify metadata
  if (subRegion->GetLargestPossibleRegion() != contiguousRegion)
  {
    std::cerr << "Region mismatch!" << std::endl;
    return EXIT_FAILURE;
  }
  if (subRegion->GetSpacing() != input->GetSpacing())
  {
    std::cerr << "Spacing mismatch!" << std::endl;
    return EXIT_FAILURE;
  }

  // Verify zero-copy: buffer pointer should point into the input's buffer
  const PixelType * inputBuf = input->GetBufferPointer();
  const PixelType * outputBuf = subRegion->GetBufferPointer();
  ptrdiff_t         offset = outputBuf - inputBuf;
  // For a contiguous extraction starting at z=2 with slice size 4*5=20:
  // offset should be 2*20 = 40
  ptrdiff_t expectedOffset = 2 * 4 * 5;
  if (offset != expectedOffset)
  {
    std::cerr << "Zero-copy FAILED: expected offset " << expectedOffset << ", got " << offset << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Zero-copy: buffer offset = " << offset << " (correct)" << std::endl;

  // Verify pixel values are accessible and correct
  itk::ImageRegionIterator<ImageType> outIt(subRegion, subRegion->GetLargestPossibleRegion());
  bool                                valuesCorrect = true;
  for (outIt.GoToBegin(); !outIt.IsAtEnd(); ++outIt)
  {
    IndexType idx = outIt.GetIndex();
    PixelType expected = static_cast<PixelType>(idx[0] + idx[1] * 10 + idx[2] * 100);
    if (std::abs(outIt.Get() - expected) > 1e-6f)
    {
      std::cerr << "Value mismatch at " << idx << ": got " << outIt.Get() << ", expected " << expected << std::endl;
      valuesCorrect = false;
      break;
    }
  }
  if (!valuesCorrect)
    return EXIT_FAILURE;
  std::cout << "Pixel values: correct" << std::endl;

  // ===== Case 2: Non-contiguous extraction (middle slice) =====
  std::cout << "\n\n****** Case 2: non-contiguous extraction (copy) ******" << std::endl;

  RegionType nonContiguousRegion;
  nonContiguousRegion.SetIndex(itk::MakeIndex(1, 1, 3));
  nonContiguousRegion.SetSize(itk::MakeSize(2, 2, 2));

  ImageType::Pointer subRegion2 = rtk::ExtractImageSubRegion(input.GetPointer(), nonContiguousRegion);

  // Verify metadata
  if (subRegion2->GetLargestPossibleRegion() != nonContiguousRegion)
  {
    std::cerr << "Region mismatch!" << std::endl;
    return EXIT_FAILURE;
  }

  // Verify zero-copy does NOT apply: buffer pointers should differ
  const PixelType * outputBuf2 = subRegion2->GetBufferPointer();
  if (outputBuf2 >= inputBuf && outputBuf2 < inputBuf + input->GetLargestPossibleRegion().GetNumberOfPixels())
  {
    std::cerr << "Non-contiguous extraction should NOT share the input buffer!" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Buffer: independent copy (correct)" << std::endl;

  // Verify pixel values via reference image
  auto reference = rtk::ConstantImageSource<ImageType>::New();
  reference->SetOrigin(itk::MakePoint(0., 0., 0.));
  reference->SetSpacing(itk::MakeVector(1., 1., 1.));
  reference->SetSize(itk::MakeSize(2, 2, 2));
  reference->SetConstant(0.f);
  reference->UpdateLargestPossibleRegion();

  // Fill reference with expected values
  itk::ImageRegionIterator<ImageType> refIt(reference->GetOutput(), reference->GetOutput()->GetLargestPossibleRegion());
  for (refIt.GoToBegin(); !refIt.IsAtEnd(); ++refIt)
  {
    IndexType idx = refIt.GetIndex();
    // Map back to input coordinates
    PixelType expected = static_cast<PixelType>((idx[0] + 1) + (idx[1] + 1) * 10 + (idx[2] + 3) * 100);
    refIt.Set(expected);
  }

  CheckImageQuality<ImageType>(subRegion2, reference->GetOutput(), 0.001, 120, 432.f);
  std::cout << "Pixel values: correct" << std::endl;

  // ===== Case 3: No buffer yet (GenerateOutputInformation scenario) =====
  std::cout << "\n\n****** Case 3: metadata-only (no buffer) ******" << std::endl;

  auto noBufferSource = rtk::ConstantImageSource<ImageType>::New();
  noBufferSource->SetOrigin(itk::MakePoint(0., 0., 0.));
  noBufferSource->SetSpacing(itk::MakeVector(1., 1., 1.));
  noBufferSource->SetSize(itk::MakeSize(4, 5, 6));
  noBufferSource->SetConstant(0.f);
  // Don't call Update — buffer is not allocated
  noBufferSource->UpdateOutputInformation();

  RegionType metaRegion;
  metaRegion.SetIndex(itk::MakeIndex(0, 0, 1));
  metaRegion.SetSize(itk::MakeSize(4, 5, 2));

  // Should not crash — returns metadata-only image
  ImageType::Pointer metaSub = rtk::ExtractImageSubRegion(noBufferSource->GetOutput(), metaRegion);
  if (metaSub->GetLargestPossibleRegion() != metaRegion)
  {
    std::cerr << "Metadata region mismatch!" << std::endl;
    return EXIT_FAILURE;
  }
#ifndef USE_CUDA
  // For CPU images, buffer should be null when input has no data.
  // For CudaImage, GetBufferPointer() has side effects on CudaDataManager
  // that prevent a clean nullptr check, so we skip this for CUDA.
  if (metaSub->GetBufferPointer() != nullptr)
  {
    std::cerr << "Expected null buffer for metadata-only image!" << std::endl;
    return EXIT_FAILURE;
  }
#endif
  std::cout << "Metadata-only: correct" << std::endl;

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
