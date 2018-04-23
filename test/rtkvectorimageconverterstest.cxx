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

int main(int, char** )
{
  typedef float                                 PixelType;
  typedef itk::Image< PixelType, 3 >            HigherDimensionImageType;
  typedef itk::Image< PixelType, 2 >            ImageType;
  typedef itk::VectorImage< PixelType, 2 >      VectorImageType;

  std::cout << "\n\n****** Case 1: 3D image to 2D vector image, and back ******" << std::endl;

  // Initialize the first input image
  HigherDimensionImageType::Pointer higherDimensionInput = HigherDimensionImageType::New();
  HigherDimensionImageType::SizeType hsize;
  HigherDimensionImageType::IndexType hindex;
  HigherDimensionImageType::SpacingType hspacing;
  HigherDimensionImageType::RegionType hregion;

  hindex.Fill(0);
  hspacing.Fill(1);
  hsize[0] = 8;
  hsize[1] = 8;
  hsize[2] = 8;
  hregion.SetSize(hsize);
  hregion.SetIndex(hindex);
  higherDimensionInput->SetSpacing( hspacing );
  higherDimensionInput->SetRegions( hregion );
  higherDimensionInput->Allocate();

  // Set each pixel's value to its position on the third dimension
  HigherDimensionImageType::RegionType slice;
  slice.SetSize(hsize);
  slice.SetSize(2, 1);
  for (unsigned int s=0; s<hsize[2]; s++)
    {
    slice.SetIndex(hindex);
    slice.SetIndex(2, s);
    itk::ImageRegionIterator<HigherDimensionImageType> sliceIt(higherDimensionInput, slice);
    while(!sliceIt.IsAtEnd())
      {
      sliceIt.Set(s);
      ++sliceIt;
      }
    }

  // Create the filter to convert it to a vector image
  typedef rtk::ImageToVectorImageFilter<HigherDimensionImageType, VectorImageType> HigherDimensionToVectorType;
  HigherDimensionToVectorType::Pointer higherDimensionToVector = HigherDimensionToVectorType::New();
  higherDimensionToVector->SetInput(higherDimensionInput);

  // Perform conversion
  TRY_AND_EXIT_ON_ITK_EXCEPTION( higherDimensionToVector->Update() );

  // Create a reference vector image, equal to the expected output of higherDimensionToVector, and perform the comparison
  VectorImageType::Pointer refVectorImage = VectorImageType::New();

  VectorImageType::SizeType refVectorSize;
  VectorImageType::IndexType refVectorIndex;
  VectorImageType::SpacingType refVectorSpacing;
  VectorImageType::RegionType refVectorRegion;

  refVectorIndex.Fill(0);
  refVectorSpacing.Fill(1);
  refVectorSize[0] = hsize[0];
  refVectorSize[1] = hsize[1];
  refVectorRegion.SetSize(refVectorSize);
  refVectorRegion.SetIndex(refVectorIndex);
  refVectorImage->SetSpacing( refVectorSpacing );
  refVectorImage->SetRegions( refVectorRegion );
  refVectorImage->SetVectorLength(hsize[2]);
  refVectorImage->Allocate();

  // Construct a vector to fill refVectorImage
  itk::VariableLengthVector<PixelType> vector;
  vector.SetSize(hsize[2]);
  for (unsigned int i=0; i<hsize[2]; i++)
    vector[i]=i;
  itk::ImageRegionIterator<VectorImageType> vecIt(refVectorImage, refVectorRegion);
  while(!vecIt.IsAtEnd())
    {
    vecIt.Set(vector);
    ++vecIt;
    }

  // Perform comparison
  CheckVariableLengthVectorImageQuality<VectorImageType>(higherDimensionToVector->GetOutput(), refVectorImage, 1e-7, 100, 2.0);

  // Create the filter to convert the result back to an image
  typedef rtk::VectorImageToImageFilter<VectorImageType, HigherDimensionImageType> VectorToHigherDimensionType;
  VectorToHigherDimensionType::Pointer vectorToHigherDimension = VectorToHigherDimensionType::New();
  vectorToHigherDimension->SetInput(higherDimensionToVector->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION( vectorToHigherDimension->Update() );

  // Compare with the initial image
  CheckImageQuality<HigherDimensionImageType>(vectorToHigherDimension->GetOutput(), higherDimensionInput, 1e-7, 100, 2.0);

  std::cout << "\n\n****** Case 2: 2D image to 2D vector image, and back ******" << std::endl;

  // Initialize the first input image
  ImageType::Pointer input = ImageType::New();
  ImageType::SizeType size;
  ImageType::IndexType index;
  ImageType::SpacingType spacing;
  ImageType::RegionType region;

  index.Fill(0);
  spacing.Fill(1);
  size[0] = 8;
  size[1] = 64;
  region.SetSize(size);
  region.SetIndex(index);
  input->SetSpacing( spacing );
  input->SetRegions( region );
  input->Allocate();

  // Set each pixel's value to its position on the third dimension
  ImageType::RegionType block;
  block.SetSize(size);
  block.SetSize(1, 8);
  for (unsigned int b=0; b<8; b++)
    {
    block.SetIndex(index);
    block.SetIndex(1, 8*b);
    itk::ImageRegionIterator<ImageType> blockIt(input, block);
    while(!blockIt.IsAtEnd())
      {
      blockIt.Set(b);
      ++blockIt;
      }
    }

  // Create the filter to convert it to a vector image
  typedef rtk::ImageToVectorImageFilter<ImageType, VectorImageType> ToVectorType;
  ToVectorType::Pointer imageToVector = ToVectorType::New();
  imageToVector->SetInput(input);
  imageToVector->SetNumberOfChannels(8);

  // Perform conversion
  TRY_AND_EXIT_ON_ITK_EXCEPTION( imageToVector->Update() );

  // Perform comparison
  CheckVariableLengthVectorImageQuality<VectorImageType>(imageToVector->GetOutput(), refVectorImage, 1e-7, 100, 2.0);

  // Create the filter to convert the result back to an image
  typedef rtk::VectorImageToImageFilter<VectorImageType, ImageType> VectorToType;
  VectorToType::Pointer vectorToImage = VectorToType::New();
  vectorToImage->SetInput(imageToVector->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION( vectorToImage->Update() );

  // Compare with the initial image
  CheckImageQuality<ImageType>(vectorToImage->GetOutput(), input, 1e-7, 100, 2.0);

  std::cout << "Test PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
