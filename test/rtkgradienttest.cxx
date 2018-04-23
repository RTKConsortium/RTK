#include <itkImageRegionConstIterator.h>
#include "rtkMacro.h"

#include "rtkTestConfiguration.h"
#include "itkRandomImageSource.h"
#include "rtkForwardDifferenceGradientImageFilter.h"

template<class TImage, class TGradient>
#if FAST_TESTS_NO_CHECKS
void CheckGradient(typename TImage::Pointer itkNotUsed(im), typename TGradient::Pointer itkNotUsed(grad), bool* itkNotUsed(dimensionsProcessed))
{
}
#else
void CheckGradient(typename TImage::Pointer im, typename TGradient::Pointer grad, bool* dimensionsProcessed)
{
  // Generate a list of indices of the dimensions to process
  std::vector<int> dimsToProcess;
  for (int dim = 0; dim < TImage::ImageDimension; dim++)
    {
    if(dimensionsProcessed[dim]) dimsToProcess.push_back(dim);
    }

  typedef itk::ImageRegionConstIterator<TGradient> GradientIteratorType;
  GradientIteratorType itGrad( grad, grad->GetBufferedRegion() );

  const int ImageDimension = TImage::ImageDimension;

  itk::Size<ImageDimension> radius;
  radius.Fill(1);

  itk::ConstNeighborhoodIterator<TImage> iit(radius, im, im->GetLargestPossibleRegion());
  itk::ZeroFluxNeumannBoundaryCondition<TImage>* boundaryCondition = new itk::ZeroFluxNeumannBoundaryCondition<TImage>;
  iit.OverrideBoundaryCondition(boundaryCondition);

  itk::SizeValueType c = (itk::SizeValueType) (iit.Size() / 2); // get offset of center pixel
  itk::SizeValueType* strides = new itk::SizeValueType[ImageDimension]; // get offsets to access neighboring pixels
  for (int dim=0; dim<ImageDimension; dim++)
    {
    strides[dim] = iit.GetStride(dim);
    }

  double AbsDiff;
  double epsilon = 1e-7;

  // Run through the image
  while(!iit.IsAtEnd())
    {
    for ( std::vector<int>::size_type k = 0; k < dimsToProcess.size(); k++ )
      {
      AbsDiff = vnl_math_abs(iit.GetPixel(c + strides[dimsToProcess[k]]) - iit.GetPixel(c) - itGrad.Get()[k] * grad->GetSpacing()[k]);

      // Checking results
      if (AbsDiff > epsilon)
        {
        std::cerr << "Test Failed: output of gradient filter not equal to finite difference image." << std::endl;
        std::cerr << "Problem at pixel " << iit.GetIndex() << std::endl;
        std::cerr << "Absolute difference = " << AbsDiff << " instead of " << epsilon << std::endl;
        exit( EXIT_FAILURE);
        }
      }
    ++iit;
    ++itGrad;
    }

}
#endif

/**
 * \file rtkgradienttest.cxx
 *
 * \brief Tests whether the gradient filter behaves as expected
 *
 * Tests whether the gradient filter compute uncentered forward
 * differences with ZeroFluxNeumann boundary conditions. The exact
 * definition of the gradient desired can be found in
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

  typedef itk::CovariantVector<OutputPixelType, 2> CovVecType;
#ifdef USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
  typedef itk::CudaImage<CovVecType, Dimension> GradientImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef itk::Image<CovVecType, Dimension> GradientImageType;
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

  // Update all sources
  TRY_AND_EXIT_ON_ITK_EXCEPTION( randomVolumeSource->Update() );

  typedef rtk::ForwardDifferenceGradientImageFilter<OutputImageType, OutputPixelType, OutputPixelType, GradientImageType> GradientFilterType;
  GradientFilterType::Pointer grad = GradientFilterType::New();
  grad->SetInput(randomVolumeSource->GetOutput());

  bool computeGradientAlongDim[Dimension];
  computeGradientAlongDim[0] = true;
  computeGradientAlongDim[1] = false;
  computeGradientAlongDim[2] = true;

  grad->SetDimensionsProcessed(computeGradientAlongDim);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( grad->Update() );

  CheckGradient<OutputImageType, GradientFilterType::OutputImageType>(randomVolumeSource->GetOutput(), grad->GetOutput(), computeGradientAlongDim);
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
