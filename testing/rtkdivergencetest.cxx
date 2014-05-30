#include <itkImageRegionConstIterator.h>
#include <itkRandomImageSource.h>
#include <itkConstantBoundaryCondition.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include "rtkMacro.h"
#include "rtkTestConfiguration.h"

#include "rtkBackwardDifferenceDivergenceImageFilter.h"
#include "rtkForwardDifferenceGradientImageFilter.h"

//#ifdef USE_CUDA
//#else
//#endif

template<class TGradient, class TImage>
#if FAST_TESTS_NO_CHECKS
void CheckScalarProducts(typename TGradient::Pointer itkNotUsed(grad1),
                         typename TGradient::Pointer itkNotUsed(grad2),
                         typename TImage::Pointer itkNotUsed(im1),
                         typename TImage::Pointer itkNotUsed(im2),
                         bool* itkNotUsed(dimensionsProcessed))
{
}
#else
void CheckScalarProducts(typename TGradient::Pointer grad1,
                         typename TGradient::Pointer grad2,
                         typename TImage::Pointer im1,
                         typename TImage::Pointer im2,
                         bool* dimensionsProcessed)
{
  // Generate a list of indices of the dimensions to process
  std::vector<int> dimsToProcess;
  for (int dim = 0; dim < TImage::ImageDimension; dim++)
    {
    if(dimensionsProcessed[dim]) dimsToProcess.push_back(dim);
    }

  typedef itk::ImageRegionConstIterator<TImage> ImageIteratorType;
  typedef itk::ImageRegionConstIterator<TGradient> GradientIteratorType;
  GradientIteratorType itGrad1( grad1, grad1->GetLargestPossibleRegion() );
  GradientIteratorType itGrad2( grad2, grad2->GetLargestPossibleRegion() );
  ImageIteratorType itIm1( im1, im1->GetLargestPossibleRegion() );
  ImageIteratorType itIm2( im2, im2->GetLargestPossibleRegion() );

  typename TImage::PixelType scalarProductGrads, scalarProductIms;
  scalarProductGrads = 0;
  scalarProductIms = 0;

  while( !itGrad1.IsAtEnd() )
    {
    for (int k=0; k<dimsToProcess.size(); k++)
      {
      scalarProductGrads += itGrad1.Get()[dimsToProcess[k]] * itGrad2.Get()[dimsToProcess[k]];
      }
    ++itGrad1;
    ++itGrad2;
    }

  while( !itIm1.IsAtEnd() )
    {
    scalarProductIms += itIm1.Get() * itIm2.Get();
    ++itIm1;
    ++itIm2;
    }

  // QI
  double ratio = scalarProductGrads / scalarProductIms;
  std::cout << "ratio = " << ratio << std::endl;

  // Checking results
  if (vcl_abs(ratio+1)>0.0001)
  {
    std::cerr << "Test Failed, ratio not valid! "
              << ratio+1 << " instead of 0.0001" << std::endl;
    exit( EXIT_FAILURE);
  }
}
#endif

/**
 * \file rtkdivergencetest.cxx
 *
 * \brief Tests whether the divergence filter behaves as expected
 *
 * Tests whether MINUS the divergence is the adjoint of the forward
 * difference gradient. The exact definition of
 * the divergence desired can be found in
 * Chambolle, Antonin. “An Algorithm for Total Variation
 * Minimization and Applications.” J. Math. Imaging Vis. 20,
 * no. 1–2 (January 2004): 89–97
 *
 * \author Cyril Mory
 */

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef double                                    OutputPixelType;

#ifdef USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > ImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension > ImageType;
#endif

  // Random image sources
  typedef itk::RandomImageSource< ImageType > RandomImageSourceType;
  RandomImageSourceType::Pointer randomVolumeSource1  = RandomImageSourceType::New();
  RandomImageSourceType::Pointer randomVolumeSource2  = RandomImageSourceType::New();

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
  randomVolumeSource1->SetOrigin( origin );
  randomVolumeSource1->SetSpacing( spacing );
  randomVolumeSource1->SetSize( size );
  randomVolumeSource1->SetMin( -7. );
  randomVolumeSource1->SetMax( 1. );

  randomVolumeSource2->SetOrigin( origin );
  randomVolumeSource2->SetSpacing( spacing );
  randomVolumeSource2->SetSize( size );
  randomVolumeSource2->SetMin( -3. );
  randomVolumeSource2->SetMax( 2. );

  // Update all sources
  TRY_AND_EXIT_ON_ITK_EXCEPTION( randomVolumeSource1->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( randomVolumeSource2->Update() );

  // Set the dimensions along which gradient and divergence
  // should be computed
  bool computeGradientAlongDim[Dimension];
  computeGradientAlongDim[0] = true;
  computeGradientAlongDim[1] = false;
  computeGradientAlongDim[2] = true;

  // Compute the gradient of both volumes
  typedef rtk::ForwardDifferenceGradientImageFilter<ImageType> GradientFilterType;
  GradientFilterType::Pointer grad1 = GradientFilterType::New();
  grad1->SetInput(randomVolumeSource1->GetOutput());
  grad1->SetDimensionsProcessed(computeGradientAlongDim);

  GradientFilterType::Pointer grad2 = GradientFilterType::New();
  grad2->SetInput(randomVolumeSource2->GetOutput());
  grad2->SetDimensionsProcessed(computeGradientAlongDim);

  // Now compute the divergence of grad2
  typedef rtk::BackwardDifferenceDivergenceImageFilter
      <GradientFilterType::OutputImageType, ImageType> DivergenceFilterType;
  DivergenceFilterType::Pointer div = DivergenceFilterType::New();
  div->SetInput(grad2->GetOutput());
  div->SetDimensionsProcessed(computeGradientAlongDim);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( div->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( grad1->Update() );

  CheckScalarProducts<GradientFilterType::OutputImageType, ImageType>
      (grad1->GetOutput(),
      grad2->GetOutput(),
      randomVolumeSource1->GetOutput(),
      div->GetOutput(),
      computeGradientAlongDim);

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
