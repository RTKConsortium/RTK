#include <itkRandomImageSource.h>
#include <itkConstantBoundaryCondition.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMultiplyImageFilter.h>

#include "rtkTest.h"
#include "rtkMacro.h"

#include "rtkBackwardDifferenceDivergenceImageFilter.h"
#include "rtkForwardDifferenceGradientImageFilter.h"

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

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = double;

  using CovVecType = itk::CovariantVector<OutputPixelType, 2>;
#ifdef USE_CUDA
  using ImageType = itk::CudaImage<OutputPixelType, Dimension>;
  using GradientImageType = itk::CudaImage<CovVecType, Dimension>;
#else
  using ImageType = itk::Image<OutputPixelType, Dimension>;
  using GradientImageType = itk::Image<CovVecType, Dimension>;
#endif

  // Random image sources
  using RandomImageSourceType = itk::RandomImageSource<ImageType>;
  auto randomVolumeSource1 = RandomImageSourceType::New();
  auto randomVolumeSource2 = RandomImageSourceType::New();

  auto origin = itk::MakePoint(-127., -127., -127.);
#if FAST_TESTS_NO_CHECKS
  auto spacing = itk::MakeVector(252., 252., 252.);
  auto size = itk::MakeSize(2, 2, 2);
#else
  auto spacing = itk::MakeVector(4., 4., 4.);
  auto size = itk::MakeSize(64, 64, 64);
#endif
  randomVolumeSource1->SetOrigin(origin);
  randomVolumeSource1->SetSpacing(spacing);
  randomVolumeSource1->SetSize(size);
  randomVolumeSource1->SetMin(-7.);
  randomVolumeSource1->SetMax(1.);

  randomVolumeSource2->SetOrigin(origin);
  randomVolumeSource2->SetSpacing(spacing);
  randomVolumeSource2->SetSize(size);
  randomVolumeSource2->SetMin(-3.);
  randomVolumeSource2->SetMax(2.);

  // Update all sources
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomVolumeSource1->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomVolumeSource2->Update());

  // Set the dimensions along which gradient and divergence
  // should be computed
  bool computeGradientAlongDim[Dimension];
  computeGradientAlongDim[0] = true;
  computeGradientAlongDim[1] = false;
  computeGradientAlongDim[2] = true;

  // Compute the gradient of both volumes
  using GradientFilterType =
    rtk::ForwardDifferenceGradientImageFilter<ImageType, OutputPixelType, OutputPixelType, GradientImageType>;
  auto grad1 = GradientFilterType::New();
  grad1->SetInput(randomVolumeSource1->GetOutput());
  grad1->SetDimensionsProcessed(computeGradientAlongDim);

  auto grad2 = GradientFilterType::New();
  grad2->SetInput(randomVolumeSource2->GetOutput());
  grad2->SetDimensionsProcessed(computeGradientAlongDim);

  // Now compute MINUS the divergence of grad2
  using DivergenceFilterType = rtk::BackwardDifferenceDivergenceImageFilter<GradientImageType, ImageType>;
  auto div = DivergenceFilterType::New();
  div->SetInput(grad2->GetOutput());
  div->SetDimensionsProcessed(computeGradientAlongDim);

  using MultiplyFilterType = itk::MultiplyImageFilter<ImageType>;
  auto multiply = MultiplyFilterType::New();
  multiply->SetInput1(div->GetOutput());
  multiply->SetConstant2(-1);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(multiply->Update());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(grad1->Update());

  CheckScalarProducts<GradientImageType, ImageType>(
    grad1->GetOutput(), grad2->GetOutput(), randomVolumeSource1->GetOutput(), multiply->GetOutput());

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
