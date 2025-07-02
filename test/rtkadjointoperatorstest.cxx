#include "rtkMacro.h"
#include "rtkTest.h"
#include "itkRandomImageSource.h"
#include "rtkConstantImageSource.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkJosephBackAttenuatedProjectionImageFilter.h"
#include "rtkJosephForwardAttenuatedProjectionImageFilter.h"
#include "rtkZengBackProjectionImageFilter.h"
#include "rtkZengForwardProjectionImageFilter.h"
#ifdef USE_CUDA
#  include "rtkCudaForwardProjectionImageFilter.h"
#  include "rtkCudaRayCastBackProjectionImageFilter.h"
#endif

/**
 * \file rtkadjointoperatorstest.cxx
 *
 * \brief Tests whether forward and back projectors are matched
 *
 * This test generates a random volume "v" and a random set of projections "p",
 * and compares the scalar products <Rv , p> and <v, R* p>, where R is either the
 * Joseph forward projector or the Cuda ray cast forward projector,
 * and R* is either the Joseph back projector or the Cuda ray cast back projector.
 * If R* is indeed the adjoint of R, these scalar products are equal.
 *
 * \author Cyril Mory
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;

#ifdef USE_CUDA
  using OutputPixelType = float;
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputPixelType = double;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 60;
#endif

  // Random image sources
  using RandomImageSourceType = itk::RandomImageSource<OutputImageType>;
  auto randomVolumeSource = RandomImageSourceType::New();
  auto randomProjectionsSource = RandomImageSourceType::New();

  // Constant sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto constantVolumeSource = ConstantImageSourceType::New();
  auto constantProjectionsSource = ConstantImageSourceType::New();
  auto constantAttenuationSource = ConstantImageSourceType::New();

  // Volume metadata
  auto origin = itk::MakePoint(-128., -128., -128.);
#if FAST_TESTS_NO_CHECKS
  auto spacing = itk::MakeVector(252., 252., 252.);
  auto size = itk::MakeSize(2, 2, 2);
#else
  auto spacing = itk::MakeVector(4., 4., 4.);
  auto size = itk::MakeSize(65, 65, 65);
#endif
  randomVolumeSource->SetOrigin(origin);
  randomVolumeSource->SetSpacing(spacing);
  randomVolumeSource->SetSize(size);
  randomVolumeSource->SetMin(0.);
  randomVolumeSource->SetMax(1.);
  randomVolumeSource->SetNumberOfWorkUnits(2); // With 1, it's deterministic

  constantVolumeSource->SetOrigin(origin);
  constantVolumeSource->SetSpacing(spacing);
  constantVolumeSource->SetSize(size);
  constantVolumeSource->SetConstant(0.);

  constantAttenuationSource->SetOrigin(origin);
  constantAttenuationSource->SetSpacing(spacing);
  constantAttenuationSource->SetSize(size);
  constantAttenuationSource->SetConstant(0.0154);

  // Projections metadata
  origin = itk::MakePoint(-128., -128., -128.);
#if FAST_TESTS_NO_CHECKS
  spacing = itk::MakeVector(504., 504., 504.);
  size = itk::MakeSize(2, 2, NumberOfProjectionImages);
#else
  spacing = itk::MakeVector(4., 4., 1.);
  size = itk::MakeSize(65, 65, NumberOfProjectionImages);
#endif
  randomProjectionsSource->SetOrigin(origin);
  randomProjectionsSource->SetSpacing(spacing);
  randomProjectionsSource->SetSize(size);
  randomProjectionsSource->SetMin(0.);
  randomProjectionsSource->SetMax(100.);

  constantProjectionsSource->SetOrigin(origin);
  constantProjectionsSource->SetSpacing(spacing);
  constantProjectionsSource->SetSize(size);
  constantProjectionsSource->SetConstant(0.);

  // Update all sources
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomVolumeSource->UpdateLargestPossibleRegion(););
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantVolumeSource->UpdateLargestPossibleRegion());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(randomProjectionsSource->UpdateLargestPossibleRegion());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantProjectionsSource->UpdateLargestPossibleRegion());

  // Geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto geometry = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);

  for (unsigned int panel = 0; panel < 2; panel++)
  {
    if (panel == 0)
      std::cout << "\n\n****** Testing with flat panel ******" << std::endl;
    else
    {
      std::cout << "\n\n****** Testing with cylindrical panel ******" << std::endl;
      geometry->SetRadiusCylindricalDetector(200);
    }

    std::cout << "\n\n****** Joseph Forward projector ******" << std::endl;

    auto fw = rtk::JosephForwardProjectionImageFilter<OutputImageType, OutputImageType>::New();
    fw->SetInput(0, constantProjectionsSource->GetOutput());
    fw->SetInput(1, randomVolumeSource->GetOutput());
    fw->SetGeometry(geometry);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(fw->Update());

    std::cout << "\n\n****** Joseph Back projector ******" << std::endl;

    auto bp = rtk::JosephBackProjectionImageFilter<OutputImageType, OutputImageType>::New();
    bp->SetInput(0, constantVolumeSource->GetOutput());
    bp->SetInput(1, randomProjectionsSource->GetOutput());
    bp->SetGeometry(geometry.GetPointer());

    TRY_AND_EXIT_ON_ITK_EXCEPTION(bp->Update());

    CheckScalarProducts<OutputImageType, OutputImageType>(
      randomVolumeSource->GetOutput(), bp->GetOutput(), randomProjectionsSource->GetOutput(), fw->GetOutput());
    std::cout << "\n\nTest PASSED! " << std::endl;

    using VectorImageType = itk::Image<itk::Vector<OutputPixelType, 3>, Dimension>;
    auto vectorRandomProjections = VectorImageType::New();
    auto vectorConstantProjections = VectorImageType::New();
    auto vectorRandomVolume = VectorImageType::New();
    auto vectorConstantVolume = VectorImageType::New();
    vectorRandomProjections->CopyInformation(randomProjectionsSource->GetOutput());
    vectorRandomProjections->SetRegions(randomProjectionsSource->GetOutput()->GetLargestPossibleRegion());
    vectorRandomProjections->Allocate();
    itk::ImageRegionIterator<VectorImageType> outIt1(vectorRandomProjections,
                                                     vectorRandomProjections->GetLargestPossibleRegion());
    itk::ImageRegionIterator<OutputImageType> inIt1(randomProjectionsSource->GetOutput(),
                                                    randomProjectionsSource->GetOutput()->GetLargestPossibleRegion());
    itk::Vector<OutputPixelType, 3>           temp;
    while (!outIt1.IsAtEnd())
    {
      temp[0] = inIt1.Get();
      temp[1] = inIt1.Get();
      temp[2] = inIt1.Get();
      outIt1.Set(temp);
      ++outIt1;
      ++inIt1;
    }

    constantVolumeSource->Update();
    vectorConstantVolume->CopyInformation(constantVolumeSource->GetOutput());
    vectorConstantVolume->SetRegions(constantVolumeSource->GetOutput()->GetLargestPossibleRegion());
    vectorConstantVolume->Allocate();
    itk::ImageRegionIterator<VectorImageType> outIt2(vectorConstantVolume,
                                                     vectorConstantVolume->GetLargestPossibleRegion());
    itk::ImageRegionIterator<OutputImageType> inIt2(constantVolumeSource->GetOutput(),
                                                    constantVolumeSource->GetOutput()->GetLargestPossibleRegion());
    while (!outIt2.IsAtEnd())
    {
      temp[0] = inIt2.Get();
      temp[1] = inIt2.Get();
      temp[2] = inIt2.Get();
      outIt2.Set(temp);
      ++outIt2;
      ++inIt2;
    }

    constantProjectionsSource->Update();
    vectorConstantProjections->CopyInformation(constantProjectionsSource->GetOutput());
    vectorConstantProjections->SetRegions(constantProjectionsSource->GetOutput()->GetLargestPossibleRegion());
    vectorConstantProjections->Allocate();
    itk::ImageRegionIterator<VectorImageType> outIt3(vectorConstantProjections,
                                                     vectorConstantProjections->GetLargestPossibleRegion());
    itk::ImageRegionIterator<OutputImageType> inIt3(constantProjectionsSource->GetOutput(),
                                                    constantProjectionsSource->GetOutput()->GetLargestPossibleRegion());
    while (!outIt3.IsAtEnd())
    {
      temp[0] = inIt3.Get();
      temp[1] = inIt3.Get();
      temp[2] = inIt3.Get();
      outIt3.Set(temp);
      ++outIt3;
      ++inIt3;
    }

    vectorRandomVolume->CopyInformation(randomVolumeSource->GetOutput());
    vectorRandomVolume->SetRegions(randomVolumeSource->GetOutput()->GetLargestPossibleRegion());
    vectorRandomVolume->Allocate();
    itk::ImageRegionIterator<VectorImageType> outIt4(vectorRandomVolume,
                                                     vectorRandomVolume->GetLargestPossibleRegion());
    itk::ImageRegionIterator<OutputImageType> inIt4(randomVolumeSource->GetOutput(),
                                                    randomVolumeSource->GetOutput()->GetLargestPossibleRegion());
    while (!outIt4.IsAtEnd())
    {
      temp[0] = inIt4.Get();
      temp[1] = inIt4.Get();
      temp[2] = inIt4.Get();
      outIt4.Set(temp);
      ++outIt4;
      ++inIt4;
    }

    std::cout << "\n\n****** Joseph Vector Forward projector ******" << std::endl;


    auto vfw = rtk::JosephForwardProjectionImageFilter<VectorImageType, VectorImageType>::New();
    vfw->SetInput(0, vectorConstantProjections);
    vfw->SetInput(1, vectorRandomVolume);
    vfw->SetGeometry(geometry);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(vfw->Update());

    std::cout << "\n\n****** Joseph Vector Back projector ******" << std::endl;
    auto vbp = rtk::JosephBackProjectionImageFilter<VectorImageType, VectorImageType>::New();
    vbp->SetInput(0, vectorConstantVolume);
    vbp->SetInput(1, vectorRandomProjections);
    vbp->SetGeometry(geometry.GetPointer());

    TRY_AND_EXIT_ON_ITK_EXCEPTION(vbp->Update());

    CheckVectorScalarProducts<VectorImageType, VectorImageType>(
      vectorRandomVolume, vbp->GetOutput(), vectorRandomProjections, vfw->GetOutput());
    std::cout << "\n\nTest PASSED! " << std::endl;

    std::cout << "\n\n****** Attenuated Joseph Forward projector ******" << std::endl;

    auto attfw = rtk::JosephForwardAttenuatedProjectionImageFilter<OutputImageType, OutputImageType>::New();
    attfw->SetInput(0, constantProjectionsSource->GetOutput());
    attfw->SetInput(1, randomVolumeSource->GetOutput());
    attfw->SetInput(2, constantAttenuationSource->GetOutput());
    attfw->SetGeometry(geometry);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(attfw->Update());

    std::cout << "\n\n****** Attenuated Joseph Back projector ******" << std::endl;

    auto attbp = rtk::JosephBackAttenuatedProjectionImageFilter<OutputImageType, OutputImageType>::New();
    attbp->SetInput(0, constantVolumeSource->GetOutput());
    attbp->SetInput(1, randomProjectionsSource->GetOutput());
    attbp->SetInput(2, constantAttenuationSource->GetOutput());
    attbp->SetGeometry(geometry.GetPointer());

    TRY_AND_EXIT_ON_ITK_EXCEPTION(attbp->Update());

    CheckScalarProducts<OutputImageType, OutputImageType>(
      randomVolumeSource->GetOutput(), attbp->GetOutput(), randomProjectionsSource->GetOutput(), attfw->GetOutput());
    std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
    std::cout << "\n\n****** Cuda Ray Cast Forward projector ******" << std::endl;

    auto cfw = rtk::CudaForwardProjectionImageFilter<OutputImageType, OutputImageType>::New();
    cfw->SetInput(0, constantProjectionsSource->GetOutput());
    cfw->SetInput(1, randomVolumeSource->GetOutput());
    cfw->SetGeometry(geometry);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(cfw->Update());

    std::cout << "\n\n****** Cuda Ray Cast Back projector ******" << std::endl;

    auto cbp = rtk::CudaRayCastBackProjectionImageFilter::New();
    cbp->SetInput(0, constantVolumeSource->GetOutput());
    cbp->SetInput(1, randomProjectionsSource->GetOutput());
    cbp->SetGeometry(geometry.GetPointer());
    cbp->SetNormalize(false);

    TRY_AND_EXIT_ON_ITK_EXCEPTION(cbp->Update());

    CheckScalarProducts<OutputImageType, OutputImageType>(
      randomVolumeSource->GetOutput(), cbp->GetOutput(), randomProjectionsSource->GetOutput(), cfw->GetOutput());
    std::cout << "\n\nTest PASSED! " << std::endl;
#endif
  }

  // Geometry parallel object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto geometry_parallel = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry_parallel->AddProjection(500., 0., noProj * 360. / NumberOfProjectionImages, 8., -4.);


  std::cout << "\n\n******  Zeng Forward projector ******" << std::endl;

  using ZengForwardProjectorType = rtk::ZengForwardProjectionImageFilter<OutputImageType, OutputImageType>;
  auto zfw = ZengForwardProjectorType::New();
  zfw->SetInput(0, constantProjectionsSource->GetOutput());
  zfw->SetInput(1, randomVolumeSource->GetOutput());
  zfw->SetGeometry(geometry_parallel);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(zfw->Update());

  std::cout << "\n\n****** Zeng Back projector ******" << std::endl;

  using ZengBackProjectorType = rtk::ZengBackProjectionImageFilter<OutputImageType, OutputImageType>;
  auto zbp = ZengBackProjectorType::New();
  zbp->SetInput(0, constantVolumeSource->GetOutput());
  zbp->SetInput(1, randomProjectionsSource->GetOutput());
  zbp->SetGeometry(geometry_parallel);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(zbp->Update());

  CheckScalarProducts<OutputImageType, OutputImageType>(
    randomVolumeSource->GetOutput(), zbp->GetOutput(), randomProjectionsSource->GetOutput(), zfw->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n******  Attenuated Zeng Forward projector ******" << std::endl;

  using ZengForwardProjectorType = rtk::ZengForwardProjectionImageFilter<OutputImageType, OutputImageType>;
  auto attzfw = ZengForwardProjectorType::New();
  attzfw->SetInput(0, constantProjectionsSource->GetOutput());
  attzfw->SetInput(1, randomVolumeSource->GetOutput());
  attzfw->SetInput(2, constantAttenuationSource->GetOutput());
  attzfw->SetGeometry(geometry_parallel);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(attzfw->Update());

  std::cout << "\n\n****** Attenuated Zeng Back projector ******" << std::endl;

  using ZengBackProjectorType = rtk::ZengBackProjectionImageFilter<OutputImageType, OutputImageType>;
  auto attzbp = ZengBackProjectorType::New();
  attzbp->SetInput(0, constantVolumeSource->GetOutput());
  attzbp->SetInput(1, randomProjectionsSource->GetOutput());
  attzbp->SetInput(2, constantAttenuationSource->GetOutput());
  attzbp->SetGeometry(geometry_parallel);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(attzbp->Update());

  CheckScalarProducts<OutputImageType, OutputImageType>(
    randomVolumeSource->GetOutput(), attzbp->GetOutput(), randomProjectionsSource->GetOutput(), attzfw->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
