#include "math.h"
#include <itkImageRegionConstIterator.h>

#include "rtkMacro.h"

#include "rtkTestConfiguration.h"
#include "itkRandomImageSource.h"
#include "rtkForwardDifferenceGradientImageFilter.h"

template <class TImage, class TGradient>
#if FAST_TESTS_NO_CHECKS
void
CheckGradient(typename TImage::Pointer    itkNotUsed(im),
              typename TGradient::Pointer itkNotUsed(grad),
              bool *                      itkNotUsed(dimensionsProcessed))
{}
#else
void
CheckGradient(typename TImage::Pointer im, typename TGradient::Pointer grad, const bool * dimensionsProcessed)
{
  // Generate a list of indices of the dimensions to process
  std::vector<int> dimsToProcess;
  for (unsigned int dim = 0; dim < TImage::ImageDimension; dim++)
  {
    if (dimensionsProcessed[dim])
      dimsToProcess.push_back(dim);
  }

  using GradientIteratorType = itk::ImageRegionConstIterator<TGradient>;
  GradientIteratorType itGrad(grad, grad->GetBufferedRegion());

  const int ImageDimension = TImage::ImageDimension;

  itk::Size<ImageDimension> radius;
  radius.Fill(1);

  itk::ConstNeighborhoodIterator<TImage> iit(radius, im, im->GetLargestPossibleRegion());
  auto *                                 boundaryCondition = new itk::ZeroFluxNeumannBoundaryCondition<TImage>;
  iit.OverrideBoundaryCondition(boundaryCondition);

  auto   c = (itk::SizeValueType)(iit.Size() / 2);         // get offset of center pixel
  auto * strides = new itk::SizeValueType[ImageDimension]; // get offsets to access neighboring pixels
  for (int dim = 0; dim < ImageDimension; dim++)
  {
    strides[dim] = iit.GetStride(dim);
  }

  double AbsDiff = NAN;
  double epsilon = 1e-7;

  // Run through the image
  while (!iit.IsAtEnd())
  {
    for (std::vector<int>::size_type k = 0; k < dimsToProcess.size(); k++)
    {
      AbsDiff = itk::Math::abs(iit.GetPixel(c + strides[dimsToProcess[k]]) - iit.GetPixel(c) -
                               itGrad.Get()[k] * grad->GetSpacing()[k]);

      // Checking results
      if (AbsDiff > epsilon)
      {
        std::cerr << "Test Failed: output of gradient filter not equal to finite difference image." << std::endl;
        std::cerr << "Problem at pixel " << iit.GetIndex() << std::endl;
        std::cerr << "Absolute difference = " << AbsDiff << " instead of " << epsilon << std::endl;
        exit(EXIT_FAILURE);
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

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = double;

  using CovVecType = itk::CovariantVector<OutputPixelType, 2>;
#ifdef USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
  using GradientImageType = itk::CudaImage<CovVecType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  using GradientImageType = itk::Image<CovVecType, Dimension>;
#endif

  // Random image sources
  using RandomImageSourceType = itk::RandomImageSource<OutputImageType>;
  auto randomVolumeSource = RandomImageSourceType::New();

  // Volume metadata
  auto origin = itk::MakePoint<OutputPixelType>(-127., -127., -127.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize<itk::SizeValueType>(2, 2, 2);
  auto spacing = itk::MakeVector<OutputPixelType>(252., 252., 252.);
#else
  auto size = itk::MakeSize<itk::SizeValueType>(64, 64, 64);
  auto spacing = itk::MakeVector<OutputPixelType>(4., 4., 4.);
#endif
  randomVolumeSource->SetOrigin(origin);
  randomVolumeSource->SetSpacing(spacing);
  randomVolumeSource->SetSize(size);
  randomVolumeSource->SetMin(0.);
  randomVolumeSource->SetMax(1.);

  // Update all sources
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomVolumeSource->Update());

  using GradientFilterType =
    rtk::ForwardDifferenceGradientImageFilter<OutputImageType, OutputPixelType, OutputPixelType, GradientImageType>;
  auto grad = GradientFilterType::New();
  grad->SetInput(randomVolumeSource->GetOutput());

  bool computeGradientAlongDim[Dimension];
  computeGradientAlongDim[0] = true;
  computeGradientAlongDim[1] = false;
  computeGradientAlongDim[2] = true;

  grad->SetDimensionsProcessed(computeGradientAlongDim);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(grad->Update());

  CheckGradient<OutputImageType, GradientFilterType::OutputImageType>(
    randomVolumeSource->GetOutput(), grad->GetOutput(), computeGradientAlongDim);
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
