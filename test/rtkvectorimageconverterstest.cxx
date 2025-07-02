#include <itksys/SystemTools.hxx>

#include "rtkTest.h"
#include "rtkConstantImageSource.h"
#include "rtkImageToVectorImageFilter.h"
#include "rtkVectorImageToImageFilter.h"

#include <itkImageRegionIterator.h>

/**
 * \file rtkvectorimageconverterstest.cxx
 *
 * \brief Test for the rtkImageToVectorImageFilter and the rtkVectorImageToImageFilter
 *
 * This code tests both behaviors of the ImageToVectorImage and VectorImageToImage filters:
 * - generates a 3D image and converts it to a 2D vector image,
 * then back to a 3D image
 * - generates a 2D image and converts it to a 2D vector image,
 * then back to a 2D image
 *
 * \author Cyril Mory
 */

int
main(int, char **)
{
  using PixelType = float;
  using HigherDimensionImageType = itk::Image<PixelType, 3>;
  using ImageType = itk::Image<PixelType, 2>;
  using VectorImageType = itk::VectorImage<PixelType, 2>;

  std::cout << "\n\n****** Case 1: 3D image to 2D vector image, and back ******" << std::endl;

  // Initialize the first input image
  auto                                 higherDimensionInput = HigherDimensionImageType::New();
  HigherDimensionImageType::RegionType hregion;

  auto hspacing = itk::MakeVector(1., 1., 1.);
  auto hsize = itk::MakeSize(8, 8, 8);
  hregion.SetSize(hsize);
  higherDimensionInput->SetSpacing(hspacing);
  higherDimensionInput->SetRegions(hregion);
  higherDimensionInput->Allocate();

  // Set each pixel's value to its position on the third dimension
  HigherDimensionImageType::RegionType slice;
  slice.SetSize(hsize);
  slice.SetSize(2, 1);
  for (unsigned int s = 0; s < hsize[2]; s++)
  {
    slice.SetIndex(2, s);
    itk::ImageRegionIterator<HigherDimensionImageType> sliceIt(higherDimensionInput, slice);
    while (!sliceIt.IsAtEnd())
    {
      sliceIt.Set(s);
      ++sliceIt;
    }
  }

  // Create the filter to convert it to a vector image
  auto higherDimensionToVector = rtk::ImageToVectorImageFilter<HigherDimensionImageType, VectorImageType>::New();
  higherDimensionToVector->SetInput(higherDimensionInput);

  // Perform conversion
  TRY_AND_EXIT_ON_ITK_EXCEPTION(higherDimensionToVector->Update());

  // Create a reference vector image, equal to the expected output of higherDimensionToVector, and perform the
  // comparison
  auto refVectorImage = VectorImageType::New();

  VectorImageType::SizeType    refVectorSize;
  VectorImageType::IndexType   refVectorIndex;
  VectorImageType::SpacingType refVectorSpacing;
  VectorImageType::RegionType  refVectorRegion;

  refVectorIndex.Fill(0);
  refVectorSpacing.Fill(1);
  refVectorSize[0] = hsize[0];
  refVectorSize[1] = hsize[1];
  refVectorRegion.SetSize(refVectorSize);
  refVectorRegion.SetIndex(refVectorIndex);
  refVectorImage->SetSpacing(refVectorSpacing);
  refVectorImage->SetRegions(refVectorRegion);
  refVectorImage->SetVectorLength(hsize[2]);
  refVectorImage->Allocate();

  // Construct a vector to fill refVectorImage
  itk::VariableLengthVector<PixelType> vector;
  vector.SetSize(hsize[2]);
  for (unsigned int i = 0; i < hsize[2]; i++)
    vector[i] = i;
  itk::ImageRegionIterator<VectorImageType> vecIt(refVectorImage, refVectorRegion);
  while (!vecIt.IsAtEnd())
  {
    vecIt.Set(vector);
    ++vecIt;
  }

  // Perform comparison
  CheckVariableLengthVectorImageQuality<VectorImageType>(
    higherDimensionToVector->GetOutput(), refVectorImage, 1e-7, 100, 2.0);

  // Create the filter to convert the result back to an image
  auto vectorToHigherDimension = rtk::VectorImageToImageFilter<VectorImageType, HigherDimensionImageType>::New();
  vectorToHigherDimension->SetInput(higherDimensionToVector->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION(vectorToHigherDimension->Update());

  // Compare with the initial image
  CheckImageQuality<HigherDimensionImageType>(
    vectorToHigherDimension->GetOutput(), higherDimensionInput, 1e-7, 100, 2.0);

  std::cout << "\n\n****** Case 2: 2D image to 2D vector image, and back ******" << std::endl;

  // Initialize the first input image
  auto                  input = ImageType::New();
  ImageType::RegionType region;

  auto spacing = itk::MakeVector(1., 1.);
  auto size = itk::MakeSize(8, 64);
  region.SetSize(size);
  input->SetSpacing(spacing);
  input->SetRegions(region);
  input->Allocate();

  // Set each pixel's value to its position on the third dimension
  ImageType::RegionType block;
  block.SetSize(size);
  block.SetSize(1, 8);
  for (unsigned int b = 0; b < 8; b++)
  {
    block.SetIndex(1, 8 * b);
    itk::ImageRegionIterator<ImageType> blockIt(input, block);
    while (!blockIt.IsAtEnd())
    {
      blockIt.Set(b);
      ++blockIt;
    }
  }

  // Create the filter to convert it to a vector image
  auto imageToVector = rtk::ImageToVectorImageFilter<ImageType, VectorImageType>::New();
  imageToVector->SetInput(input);
  imageToVector->SetNumberOfChannels(8);

  // Perform conversion
  TRY_AND_EXIT_ON_ITK_EXCEPTION(imageToVector->Update());

  // Perform comparison
  CheckVariableLengthVectorImageQuality<VectorImageType>(imageToVector->GetOutput(), refVectorImage, 1e-7, 100, 2.0);

  // Create the filter to convert the result back to an image
  auto vectorToImage = rtk::VectorImageToImageFilter<VectorImageType, ImageType>::New();
  vectorToImage->SetInput(imageToVector->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION(vectorToImage->Update());

  // Compare with the initial image
  CheckImageQuality<ImageType>(vectorToImage->GetOutput(), input, 1e-7, 100, 2.0);

  std::cout << "Test PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
