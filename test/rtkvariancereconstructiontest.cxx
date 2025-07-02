#include <itkImageRegionConstIterator.h>
#include <itkAddImageFilter.h>
#include <itkSquareImageFilter.h>
#include <itkMultiplyImageFilter.h>

#include "rtkTestConfiguration.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"
#include "itkShotNoiseImageFilter.h"
#include "rtkTest.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkFDKVarianceReconstructionFilter.h"

/**
 * \file rtkvariancereconstructiontest.cxx
 *
 * \brief Functional test for the filter that reconstruct the variance map with FDK.
 *
 * This test generates a set of noisy projections of a Shepp-Logan phantom and
 * reconstruct them using FDK algorithm. Then, the variance of the reconstructed
 * images is computed and compared to the variance map obtain with the tested filter.
 *
 * \author Simon Rit
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputImageType = itk::Image<float, Dimension>;
  constexpr double hann = 1.0;
#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
  constexpr unsigned int NumberOfSamples = 2;
#else
  constexpr unsigned int NumberOfProjectionImages = 10;
  constexpr unsigned int NumberOfSamples = 100;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto tomographySource = ConstantImageSourceType::New();
  auto origin = itk::MakePoint(-7.5, 0., -7.5);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(2, 2, 2);
  auto spacing = itk::MakeVector(254., 254., 254.);
#else
  auto size = itk::MakeSize(16, 1, 16);
  auto spacing = itk::MakeVector(1., 1., 1.);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  auto projectionsSource = ConstantImageSourceType::New();
  origin = itk::MakePoint(-7.5, -1., -7.5);
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(2, 2, NumberOfProjectionImages);
  spacing = itk::MakeVector(508., 508., 508.);
#else
  size = itk::MakeSize(64, 3, NumberOfProjectionImages);
  spacing = itk::MakeVector(1., 1., 1.);
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  // Geometry object
  auto geometry = rtk::ThreeDCircularProjectionGeometry::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 0., noProj * 360. / NumberOfProjectionImages);

  // Shepp Logan projections filter
  auto slp = rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType>::New();
  slp->SetInput(projectionsSource->GetOutput());
  slp->SetGeometry(geometry);
  slp->SetPhantomScale(8.);
  slp->InPlaceOff();

  std::cout << "\n\n****** Test : Create a set of noisy projections ******" << std::endl;

  // Add noise
  auto noisy = itk::ShotNoiseImageFilter<OutputImageType>::New();
  noisy->SetInput(slp->GetOutput());

  using AddType = itk::AddImageFilter<OutputImageType, OutputImageType>;
  auto add = AddType::New();
  add->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(tomographySource->Update());
  OutputImageType::Pointer currentSum = tomographySource->GetOutput();
  currentSum->DisconnectPipeline();

  auto square = itk::SquareImageFilter<OutputImageType, OutputImageType>::New();
  auto addSquare = AddType::New();
  addSquare->SetInput(1, square->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(tomographySource->Update());
  OutputImageType::Pointer currentSumOfSquares = tomographySource->GetOutput();
  currentSumOfSquares->DisconnectPipeline();

  // FDK reconstruction
  auto feldkamp = rtk::FDKConeBeamReconstructionFilter<OutputImageType>::New();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(tomographySource->Update());
  feldkamp->SetInput(0, tomographySource->GetOutput());
  feldkamp->SetInput(1, noisy->GetOutput());
  feldkamp->SetGeometry(geometry);
  feldkamp->GetRampFilter()->SetHannCutFrequency(hann);
  add->SetInput(1, feldkamp->GetOutput());
  square->SetInput(0, feldkamp->GetOutput());

  for (unsigned int i = 0; i < NumberOfSamples; ++i)
  {
    // New realization
    noisy->SetSeed(i);

    // Update sum
    add->SetInput(0, currentSum);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(add->Update());
    currentSum = add->GetOutput();
    currentSum->DisconnectPipeline();

    // Update sum of squared values
    addSquare->SetInput(0, currentSumOfSquares);
    addSquare->SetInput(1, square->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(addSquare->Update());
    currentSumOfSquares = addSquare->GetOutput();
    currentSumOfSquares->DisconnectPipeline();
  }

  auto multiply = itk::MultiplyImageFilter<OutputImageType, OutputImageType>::New();
  multiply->SetInput(0, currentSumOfSquares);
  multiply->SetConstant((float)(1. / NumberOfSamples));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(multiply->Update());
  currentSumOfSquares = multiply->GetOutput();
  currentSumOfSquares->DisconnectPipeline();
  addSquare->SetInput(0, currentSumOfSquares);

  currentSum->DisconnectPipeline();
  square->SetInput(currentSum);
  multiply->SetInput(0, square->GetOutput());
  multiply->SetConstant((float)(-1. / (NumberOfSamples * NumberOfSamples)));
  addSquare->SetInput(1, multiply->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(addSquare->Update());

  auto variance = rtk::FDKVarianceReconstructionFilter<OutputImageType, OutputImageType>::New();
  variance->SetGeometry(geometry);
  variance->SetInput(0, tomographySource->GetOutput());
  variance->SetInput(1, slp->GetOutput());
  variance->GetVarianceRampFilter()->SetHannCutFrequency(hann);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(variance->Update());

  CheckImageQuality<OutputImageType>(addSquare->GetOutput(), variance->GetOutput(), 0.72, 22.4, 2.);
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
