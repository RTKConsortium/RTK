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

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef double                                    OutputPixelType;

  typedef itk::CovariantVector<OutputPixelType, 2> CovVecType;
#ifdef USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > ImageType;
  typedef itk::CudaImage<CovVecType, Dimension> GradientImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension > ImageType;
  typedef itk::Image<CovVecType, Dimension> GradientImageType;
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
  typedef rtk::ForwardDifferenceGradientImageFilter<ImageType, OutputPixelType, OutputPixelType, GradientImageType> GradientFilterType;
  GradientFilterType::Pointer grad1 = GradientFilterType::New();
  grad1->SetInput(randomVolumeSource1->GetOutput());
  grad1->SetDimensionsProcessed(computeGradientAlongDim);

  GradientFilterType::Pointer grad2 = GradientFilterType::New();
  grad2->SetInput(randomVolumeSource2->GetOutput());
  grad2->SetDimensionsProcessed(computeGradientAlongDim);

  // Now compute MINUS the divergence of grad2
  typedef rtk::BackwardDifferenceDivergenceImageFilter
      <GradientImageType, ImageType> DivergenceFilterType;
  DivergenceFilterType::Pointer div = DivergenceFilterType::New();
  div->SetInput(grad2->GetOutput());
  div->SetDimensionsProcessed(computeGradientAlongDim);

  typedef itk::MultiplyImageFilter<ImageType> MultiplyFilterType;
  MultiplyFilterType::Pointer multiply = MultiplyFilterType::New();
  multiply->SetInput1(div->GetOutput());
  multiply->SetConstant2(-1);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( multiply->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( grad1->Update() );

  CheckScalarProducts<GradientImageType, ImageType>
      (grad1->GetOutput(),
      grad2->GetOutput(),
      randomVolumeSource1->GetOutput(),
      multiply->GetOutput());

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
