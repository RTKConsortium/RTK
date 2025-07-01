#include <itkPasteImageFilter.h>
#include <itksys/SystemTools.hxx>
#include <itkJoinSeriesImageFilter.h>
#include <itkDivideImageFilter.h>

#include "rtkTest.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"
#include "rtkWarpProjectionStackToFourDImageFilter.h"
#include "rtkPhasesToInterpolationWeights.h"
#include "rtkBackProjectionImageFilter.h"

/**
 * \file rtkwarpprojectionstacktofourdtest.cxx
 *
 * \brief Functional test for classes performing motion-compensated
 * back projection combining warping and back projection
 *
 * This test generates the projections of a phantom, which consists of two
 * ellipsoids (one of them moving). The resulting moving phantom is
 * back projected, each ray following a trajectory that matches the motion
 * The result is compared to the CPU voxel based back projection of the static object.
 *
 * \author Cyril Mory
 */

int
main(int, char **)
{
  using OutputPixelType = float;

  using DVFVectorType = itk::CovariantVector<OutputPixelType, 3>;

#ifdef USE_CUDA
  using VolumeSeriesType = itk::CudaImage<OutputPixelType, 4>;
  using ProjectionStackType = itk::CudaImage<OutputPixelType, 3>;
  using VolumeType = itk::CudaImage<OutputPixelType, 3>;
  using DVFSequenceImageType = itk::CudaImage<DVFVectorType, VolumeSeriesType::ImageDimension>;
#else
  using VolumeSeriesType = itk::Image<OutputPixelType, 4>;
  using ProjectionStackType = itk::Image<OutputPixelType, 3>;
  using VolumeType = itk::Image<OutputPixelType, 3>;
  using DVFSequenceImageType = itk::Image<DVFVectorType, VolumeSeriesType::ImageDimension>;
#endif

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 5;
#else
  constexpr unsigned int NumberOfProjectionImages = 64;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<VolumeType>;
  using FourDSourceType = rtk::ConstantImageSource<VolumeSeriesType>;

  auto tomographySource = ConstantImageSourceType::New();
  auto origin = itk::MakePoint(-63., -31., -63.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(8, 8, 8);
  auto spacing = itk::MakeVector(16., 8., 16.);
#else
  auto size = itk::MakeSize(64, 64, 64);
  auto spacing = itk::MakeVector(2., 1., 2.);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  auto fourdSource = FourDSourceType::New();
  auto fourDOrigin = itk::MakePoint(-63., -31.5, -63., 0.);
#if FAST_TESTS_NO_CHECKS
  auto fourDSize = itk::MakeSize(8, 8, 8, 2);
  auto fourDSpacing = itk::MakeVector(16., 8., 16., 1.);
#else
  auto fourDSize = itk::MakeSize(64, 64, 64, 8);
  auto fourDSpacing = itk::MakeVector(2., 1., 2., 1.);
#endif
  fourdSource->SetOrigin(fourDOrigin);
  fourdSource->SetSpacing(fourDSpacing);
  fourdSource->SetSize(fourDSize);
  fourdSource->SetConstant(0.);

  auto projectionsSource = ConstantImageSourceType::New();
  origin = itk::MakePoint(-254., -254., -254.);
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(32, 32, NumberOfProjectionImages);
  spacing = itk::MakeVector(32., 32., 32.);
#else
  size = itk::MakeSize(128, 128, NumberOfProjectionImages);
  spacing = itk::MakeVector(4., 4., 1.);
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  auto oneProjectionSource = ConstantImageSourceType::New();
  size[2] = 1;
  oneProjectionSource->SetOrigin(origin);
  oneProjectionSource->SetSpacing(spacing);
  oneProjectionSource->SetSize(size);
  oneProjectionSource->SetConstant(0.);

  for (double radius = 0; radius < 2; radius++)
  {
    // Geometry object
    using GeometryType = rtk::ThreeDCircularProjectionGeometry;
    auto geometry = GeometryType::New();

    // Projections
    using REIType = rtk::RayEllipsoidIntersectionImageFilter<VolumeType, ProjectionStackType>;
    using PasteImageFilterType = itk::PasteImageFilter<ProjectionStackType, ProjectionStackType, ProjectionStackType>;
    auto destinationIndex = itk::MakeIndex(0, 0, 0);

    auto pasteFilter = PasteImageFilterType::New();
    pasteFilter->SetDestinationImage(projectionsSource->GetOutput());

    auto pasteFilterStaticProjections = PasteImageFilterType::New();
    pasteFilterStaticProjections->SetDestinationImage(projectionsSource->GetOutput());

#ifdef USE_CUDA
    std::string signalFileName = "signal_fw_cuda.txt";
#else
    std::string signalFileName = "signal_fw.txt";
#endif

    std::ofstream signalFile(signalFileName.c_str());

    for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    {
      geometry->SetRadiusCylindricalDetector(radius * 1200.);
      geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

      // Geometry object
      auto oneProjGeometry = GeometryType::New();
      oneProjGeometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

      // Ellipse 1
      auto e1 = REIType::New();
      auto semiprincipalaxis = itk::MakeVector(60., 30., 60.);
      auto center = itk::MakeVector(0., 0., 0.);
      e1->SetInput(oneProjectionSource->GetOutput());
      e1->SetGeometry(oneProjGeometry);
      e1->SetDensity(1.);
      e1->SetAxis(semiprincipalaxis);
      e1->SetCenter(center);
      e1->SetAngle(0.);
      e1->InPlaceOff();
      e1->Update();

      // Ellipse 2
      auto e2 = REIType::New();
      semiprincipalaxis.Fill(8.);
      center[0] = 4 * (itk::Math::abs((4 + noProj) % 8 - 4.) - 2.);
      e2->SetInput(e1->GetOutput());
      e2->SetGeometry(oneProjGeometry);
      e2->SetDensity(-1.);
      e2->SetAxis(semiprincipalaxis);
      e2->SetCenter(center);
      e2->SetAngle(0.);
      e2->Update();

      // Ellipse 2 without motion
      auto e2static = REIType::New();
      center[0] = 0;
      e2static->SetInput(e1->GetOutput());
      e2static->SetGeometry(oneProjGeometry);
      e2static->SetDensity(-1.);
      e2static->SetAxis(semiprincipalaxis);
      e2static->SetCenter(center);
      e2static->SetAngle(0.);
      e2static->Update();

      // Adding each projection to the projection stacks
      if (noProj > 0) // After the first projection, we use the output as input
      {
        ProjectionStackType::Pointer wholeImage = pasteFilter->GetOutput();
        wholeImage->DisconnectPipeline();
        pasteFilter->SetDestinationImage(wholeImage);

        ProjectionStackType::Pointer wholeImageStatic = pasteFilterStaticProjections->GetOutput();
        wholeImageStatic->DisconnectPipeline();
        pasteFilterStaticProjections->SetDestinationImage(wholeImageStatic);
      }
      pasteFilter->SetSourceImage(e2->GetOutput());
      pasteFilter->SetSourceRegion(e2->GetOutput()->GetLargestPossibleRegion());
      pasteFilter->SetDestinationIndex(destinationIndex);
      pasteFilter->UpdateLargestPossibleRegion();

      pasteFilterStaticProjections->SetSourceImage(e2static->GetOutput());
      pasteFilterStaticProjections->SetSourceRegion(e2static->GetOutput()->GetLargestPossibleRegion());
      pasteFilterStaticProjections->SetDestinationIndex(destinationIndex);
      pasteFilterStaticProjections->UpdateLargestPossibleRegion();

      destinationIndex[2]++;

      // Signal
      signalFile << (noProj % 8) / 8. << std::endl;
    }
    signalFile.close();

    // Create a vector field and its (very rough) inverse
    using IteratorType = itk::ImageRegionIteratorWithIndex<DVFSequenceImageType>;

    auto deformationField = DVFSequenceImageType::New();
    auto inverseDeformationField = DVFSequenceImageType::New();

    auto sizeMotion = itk::MakeSize(fourDSize[0], fourDSize[1], fourDSize[2], 2);
    auto originMotion = itk::MakePoint(-63., -31., -63., 0.);
    auto spacingMotion = itk::MakeVector(fourDSpacing[0], fourDSpacing[1], fourDSpacing[2], fourDSpacing[3]);
    DVFSequenceImageType::RegionType regionMotion;
    regionMotion.SetSize(sizeMotion);

    deformationField->SetRegions(regionMotion);
    deformationField->SetOrigin(originMotion);
    deformationField->SetSpacing(spacingMotion);
    deformationField->Allocate();

    inverseDeformationField->SetRegions(regionMotion);
    inverseDeformationField->SetOrigin(originMotion);
    inverseDeformationField->SetSpacing(spacingMotion);
    inverseDeformationField->Allocate();

    // Vector Field initilization
    DVFVectorType vec;
    IteratorType  dvfIt(deformationField, deformationField->GetLargestPossibleRegion());
    IteratorType  idvfIt(inverseDeformationField, inverseDeformationField->GetLargestPossibleRegion());

    DVFSequenceImageType::OffsetType DVFCenter;
    DVFSequenceImageType::IndexType  toCenter;
    DVFCenter.Fill(0);
    DVFCenter[0] = sizeMotion[0] / 2;
    DVFCenter[1] = sizeMotion[1] / 2;
    DVFCenter[2] = sizeMotion[2] / 2;
    while (!dvfIt.IsAtEnd())
    {
      vec.Fill(0.);
      toCenter = dvfIt.GetIndex() - DVFCenter;

      if (0.3 * toCenter[0] * toCenter[0] + 0.5 * toCenter[1] * toCenter[1] + 0.5 * toCenter[2] * toCenter[2] < 40)
      {
        if (dvfIt.GetIndex()[3] == 0)
          vec[0] = -8.;
        else
          vec[0] = 8.;
      }
      dvfIt.Set(vec);
      idvfIt.Set(-vec);

      ++dvfIt;
      ++idvfIt;
    }

    // Input 4D volume sequence
    using JoinFilterType = itk::JoinSeriesImageFilter<VolumeType, VolumeSeriesType>;
    auto join = JoinFilterType::New();

    // Read the phases file
    auto phaseReader = rtk::PhasesToInterpolationWeights::New();
    phaseReader->SetFileName(signalFileName);
    phaseReader->SetNumberOfReconstructedFrames(fourDSize[3]);
    phaseReader->Update();

    // The CPU voxel-based static 3D back projection used as a reference for comparison
    using StaticBackProjectionFilterType = rtk::BackProjectionImageFilter<VolumeType, VolumeType>;
    auto backprojection = StaticBackProjectionFilterType::New();
    backprojection->SetInput(0, tomographySource->GetOutput());
    backprojection->SetInput(1, pasteFilterStaticProjections->GetOutput());
    backprojection->SetGeometry(geometry.GetPointer());
    backprojection->Update();

    // Divide the result by the number of volumes in the 4D sequence,
    // for the attenuation results to approximately match
    using DivideFilterType = itk::DivideImageFilter<VolumeType, VolumeType, VolumeType>;
    auto divide = DivideFilterType::New();
    divide->SetInput1(backprojection->GetOutput());
    divide->SetConstant2(fourDSize[3]);
    divide->Update();

    // Concatenate n times the 3D reconstruction with itself to compare the result with
    // the 4D output of warpbackproject
    for (itk::SizeValueType n = 0; n < fourDSize[3]; n++)
      join->SetInput(n, divide->GetOutput());
    join->Update();

    // Create and set the warped forward projection filter
    using WarpProjectionStackToFourDType = rtk::WarpProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType>;
    auto warpbackproject = WarpProjectionStackToFourDType::New();
    warpbackproject->SetInputVolumeSeries(fourdSource->GetOutput());
    warpbackproject->SetInputProjectionStack(pasteFilter->GetOutput());
    warpbackproject->SetGeometry(geometry);
    warpbackproject->SetDisplacementField(deformationField);
    warpbackproject->SetWeights(phaseReader->GetOutput());
    warpbackproject->SetSignal(rtk::ReadSignalFile(signalFileName));

#ifndef USE_CUDA
    std::cout << "\n\n****** Case " << 1 + 2 * radius
              << ": Non-warped voxel based back projection (warped back projection exists only in CUDA) ******"
              << std::endl;
    if (radius)
      std::cout << "\n\n****** Cylindrical detector ******" << std::endl;
    TRY_AND_EXIT_ON_ITK_EXCEPTION(warpbackproject->Update());
    CheckImageQuality<VolumeSeriesType>(warpbackproject->GetOutput(), join->GetOutput(), 15, 33, 800.0);
    std::cout << "\n\nTest PASSED! " << std::endl;
#endif

#ifdef USE_CUDA
    std::cout << "\n\n****** Case " << 2 + 2 * radius << ": CUDA warped back projection ******" << std::endl;
    if (radius)
      std::cout << "\n\n****** Cylindrical detector ******" << std::endl;
    TRY_AND_EXIT_ON_ITK_EXCEPTION(warpbackproject->Update());
    CheckImageQuality<VolumeSeriesType>(warpbackproject->GetOutput(), join->GetOutput(), 15, 33, 800.0);
    std::cout << "\n\nTest PASSED! " << std::endl;
#endif

    itksys::SystemTools::RemoveFile(signalFileName.c_str());
  }

  return EXIT_SUCCESS;
}
