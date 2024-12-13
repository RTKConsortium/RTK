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
  using OutputPixelType = float;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
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
  ConstantImageSourceType::PointType   origin{ itk::MakePoint(-7.5, 0., -7.5) };
  ConstantImageSourceType::SizeType    size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource = ConstantImageSourceType::New();
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(2, 2, 2);
  spacing = itk::MakeVector(254., 254., 254.);
#else
  size = itk::MakeSize(16, 1, 16);
  spacing = itk::MakeVector(1., 1., 1.);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
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
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 0., noProj * 360. / NumberOfProjectionImages);

  // Shepp Logan projections filter
  using SLPType = rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType>;
  SLPType::Pointer slp = SLPType::New();
  slp->SetInput(projectionsSource->GetOutput());
  slp->SetGeometry(geometry);
  slp->SetPhantomScale(8.);
  slp->InPlaceOff();

  std::cout << "\n\n****** Test : Create a set of noisy projections ******" << std::endl;

  // Add noise
  using NIFType = itk::ShotNoiseImageFilter<OutputImageType>;
  NIFType::Pointer noisy = NIFType::New();
  noisy->SetInput(slp->GetOutput());

  using AddType = itk::AddImageFilter<OutputImageType, OutputImageType>;
  AddType::Pointer add = AddType::New();
  add->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(tomographySource->Update());
  OutputImageType::Pointer currentSum = tomographySource->GetOutput();
  currentSum->DisconnectPipeline();

  using SquareType = itk::SquareImageFilter<OutputImageType, OutputImageType>;
  SquareType::Pointer square = SquareType::New();
  AddType::Pointer    addSquare = AddType::New();
  addSquare->SetInput(1, square->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(tomographySource->Update());
  OutputImageType::Pointer currentSumOfSquares = tomographySource->GetOutput();
  currentSumOfSquares->DisconnectPipeline();

  // FDK reconstruction
  using FDKType = rtk::FDKConeBeamReconstructionFilter<OutputImageType>;
  FDKType::Pointer feldkamp = FDKType::New();
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

  using MultiplyType = itk::MultiplyImageFilter<OutputImageType, OutputImageType>;
  MultiplyType::Pointer multiply = MultiplyType::New();
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

  using VarianceType = rtk::FDKVarianceReconstructionFilter<OutputImageType, OutputImageType>;
  VarianceType::Pointer variance = VarianceType::New();
  variance->SetGeometry(geometry);
  variance->SetInput(0, tomographySource->GetOutput());
  variance->SetInput(1, slp->GetOutput());
  variance->GetVarianceRampFilter()->SetHannCutFrequency(hann);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(variance->Update());

  CheckImageQuality<OutputImageType>(addSquare->GetOutput(), variance->GetOutput(), 0.72, 22.4, 2.);
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
