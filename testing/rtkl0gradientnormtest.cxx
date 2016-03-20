#include <itkPasteImageFilter.h>
#include <itksys/SystemTools.hxx>
#include <itkJoinSeriesImageFilter.h>

#include "rtkTest.h"
#include <itkImageRegionIteratorWithIndex.h>
#include "rtkLastDimensionL0GradientDenoisingImageFilter.h"

/**
 * \file rtkl0gradientnormtest.cxx
 *
 * \brief Test for the denoising filter that minimizes the L0 norm of the gradient
 * along the last dimension.
 *
 * This test generates a known 4D image, degrades it with noise,
 * runs L0 gradient norm denoising, and compares the L0 norm of the
 * gradient along the last dimension before and after the regularization
 *
 * \author Cyril Mory
 */

template< class TInputImage >
static unsigned int ComputeL0NormAlongLastDimension(typename TInputImage::Pointer in)
{

  // Create a slice to iterate on
  typename TInputImage::RegionType largest = in->GetLargestPossibleRegion();
  typename TInputImage::RegionType FirstFrameRegion = largest;
  FirstFrameRegion.SetSize(TInputImage::ImageDimension - 1, 1);
  itk::ImageRegionIteratorWithIndex<TInputImage> FakeIterator(in, FirstFrameRegion);

  // Useful
  unsigned int length = largest.GetSize(TInputImage::ImageDimension - 1);

  // Create a single-voxel region traversing last dimension
  typename TInputImage::RegionType SingleVoxelRegion = largest;
  for (unsigned int dim = 0; dim< TInputImage::ImageDimension - 1; dim++)
    {
    SingleVoxelRegion.SetSize(dim, 1);
    SingleVoxelRegion.SetIndex(dim, 0);
    }

  unsigned int norm = 0;
  typename TInputImage::PixelType* oned = new typename TInputImage::PixelType[length];

  // Walk the first frame, and for each voxel, set the whole sequence the way we want it
  while(!FakeIterator.IsAtEnd())
    {
    // Configure the SingleVoxelRegion correctly to follow the FakeIterator
    // It is the only purpose of this FakeIterator
    SingleVoxelRegion.SetIndex(FakeIterator.GetIndex());

    // Walk the input along last dimension for this voxel, filling it with the values we want
    itk::ImageRegionConstIterator<TInputImage> inputIterator(in, SingleVoxelRegion);

    unsigned int i=0;
    while (!inputIterator.IsAtEnd())
      {
      oned[i] = inputIterator.Get();
      i++;
      ++inputIterator;
      }

    for(unsigned int j = 0; j<length; j++)
      {
      unsigned int next = (j + 1) % length;
      if (oned[j] != oned[next]) norm++;
      i++;
      }

    ++FakeIterator;
    }
  delete [] oned;
  return(norm);
}

template<class TInputImage>
void CheckL0NormOfGradient(typename TInputImage::Pointer before, typename TInputImage::Pointer after)
{
  unsigned int normBefore;
  unsigned int normAfter;

  normBefore = ComputeL0NormAlongLastDimension<TInputImage>( before );
  std::cout << "L0 norm of the gradient before denoising is " << normBefore << std::endl;

  normAfter = ComputeL0NormAlongLastDimension<TInputImage>( after );
  std::cout << "L0 norm of the gradient after denoising is " << normAfter << std::endl;

  // Checking results
  if (normBefore/2 < normAfter)
    {
    std::cerr << "Test Failed: L0 norm of the gradient was not reduced enough" << std::endl;
    exit( EXIT_FAILURE);
    }
}



int main(int, char** )
{
  typedef float                             OutputPixelType;
  typedef itk::Image< OutputPixelType, 4 >  VolumeSeriesType;

  // Constant image sources
  VolumeSeriesType::PointType origin;
  VolumeSeriesType::SizeType size;
  VolumeSeriesType::SpacingType spacing;
  VolumeSeriesType::IndexType index;

  origin[0] = -63.;
  origin[1] = -31.;
  origin[2] = -63.;
  origin[3] = 0.;
  index.Fill(0);
#if FAST_TESTS_NO_CHECKS
  size[0] = 8;
  size[1] = 8;
  size[2] = 8;
  size[3] = 12;
  spacing[0] = 16.;
  spacing[1] = 8.;
  spacing[2] = 16.;
  spacing[3] = 1.;
#else
  size[0] = 32;
  size[1] = 16;
  size[2] = 32;
  size[3] = 12;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
  spacing[3] = 1.;
#endif

  // Initialize the largest possible region for the input image
  VolumeSeriesType::RegionType largest;
  largest.SetIndex(index);
  largest.SetSize(size);

  // Initialize the input image
  VolumeSeriesType::Pointer input = VolumeSeriesType::New();
  input->SetOrigin(origin);
  input->SetSpacing(spacing);
  input->SetRegions(largest);
  input->Allocate();

  ////////////////////////////////////////////////
  // Fill the input image with the values we want

  // Determine the values we want
  OutputPixelType* signal = new OutputPixelType[largest.GetSize(VolumeSeriesType::ImageDimension - 1)]; //Should be an array of 12 floats, since size[3] = 12
  signal[0] = 0;
  signal[1] = 1;
  signal[2] = 1;
  signal[3] = 1;
  signal[4] = 0.5;
  signal[5] = 0;
  signal[6] = -0.5;
  signal[7] = -1;
  signal[8] = -1;
  signal[9] = -1;
  signal[10] = 0.5;
  signal[11] = 0;

  // Create a slice to iterate on
  VolumeSeriesType::RegionType FirstFrameRegion = largest;
  FirstFrameRegion.SetSize(VolumeSeriesType::ImageDimension - 1, 1);
  itk::ImageRegionIteratorWithIndex<VolumeSeriesType> FakeIterator(input, FirstFrameRegion);

  // Create a single-voxel region traversing last dimension
  VolumeSeriesType::RegionType SingleVoxelRegion = largest;
  for (unsigned int dim = 0; dim< VolumeSeriesType::ImageDimension - 1; dim++)
    {
    SingleVoxelRegion.SetSize(dim, 1);
    SingleVoxelRegion.SetIndex(dim, 0);
    }

  // Walk the first frame, and for each voxel, set the whole sequence the way we want it
  while(!FakeIterator.IsAtEnd())
    {
    // Configure the SingleVoxelRegion correctly to follow the FakeIterator
    // It is the only purpose of this FakeIterator
    SingleVoxelRegion.SetIndex(FakeIterator.GetIndex());

    // Walk the input along last dimension for this voxel, filling it with the values we want
    itk::ImageRegionIterator<VolumeSeriesType> inputIterator(input, SingleVoxelRegion);

    unsigned int i=0;
    while (!inputIterator.IsAtEnd())
      {
      float randomNoise = rand() % 10000;
      randomNoise -= 5000;
      randomNoise /= 25000;

      inputIterator.Set(signal[i] + randomNoise);
      i++;
      ++inputIterator;
      }
    ++FakeIterator;
    }
  delete [] signal;

   // Perform regularization
  typedef rtk::LastDimensionL0GradientDenoisingImageFilter<VolumeSeriesType> DenoisingFilterType;
  DenoisingFilterType::Pointer denoising = DenoisingFilterType::New();
  denoising->SetInput(input);
  denoising->SetLambda(0.3);
  denoising->SetNumberOfIterations(5);
  denoising->SetInPlace(false);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( denoising->Update() );

  CheckL0NormOfGradient<VolumeSeriesType>(input, denoising->GetOutput() );

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
