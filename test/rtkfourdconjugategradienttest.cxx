#include <itkImageRegionConstIterator.h>
#include <itkPasteImageFilter.h>
#include <itksys/SystemTools.hxx>
#include <itkJoinSeriesImageFilter.h>

#include "rtkTest.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"
#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkPhasesToInterpolationWeights.h"

/**
 * \file rtkfourdconjugategradienttest.cxx
 *
 * \brief Functional test for classes performing 4D conjugate gradient-based
 * reconstruction.
 *
 * This test generates the projections of a phantom, which consists of two
 * ellipsoids (one of them moving). The resulting moving phantom is
 * reconstructed using 4D conjugate gradient and the generated
 * result is compared to the expected results (analytical computation).
 *
 * \author Cyril Mory
 */

int
main(int, char **)
{
  using OutputPixelType = float;

#ifdef USE_CUDA
  using VolumeSeriesType = itk::CudaImage<OutputPixelType, 4>;
  using ProjectionStackType = itk::CudaImage<OutputPixelType, 3>;
  using VolumeType = itk::CudaImage<OutputPixelType, 3>;
#else
  using VolumeSeriesType = itk::Image<OutputPixelType, 4>;
  using ProjectionStackType = itk::Image<OutputPixelType, 3>;
  using VolumeType = itk::Image<OutputPixelType, 3>;
#endif

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 5;
#else
  constexpr unsigned int NumberOfProjectionImages = 64;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<VolumeType>;

  auto tomographySource = ConstantImageSourceType::New();
  auto origin = itk::MakePoint(-63., -31., -63.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(8, 8, 8);
  auto spacing = itk::MakeVector(16., 8., 16.);
#else
  auto size = itk::MakeSize(32, 16, 32);
  auto spacing = itk::MakeVector(4., 4., 4.);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  auto fourdSource = rtk::ConstantImageSource<VolumeSeriesType>::New();
  auto fourDOrigin = itk::MakePoint(-63., -31., -63., 0.);
#if FAST_TESTS_NO_CHECKS
  auto fourDSize = itk::MakeSize(8, 8, 8, 2);
  auto fourDSpacing = itk::MakeVector(16., 8., 16., 1.);
#else
  auto fourDSize = itk::MakeSize(32, 16, 32, 8);
  auto fourDSpacing = itk::MakeVector(4., 4., 4., 1.);
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
  size = itk::MakeSize(64, 64, NumberOfProjectionImages);
  spacing = itk::MakeVector(8., 8., 1.);
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

  // Geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto geometry = GeometryType::New();

  // Projections
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<VolumeType, ProjectionStackType>;
  auto destinationIndex = itk::MakeIndex(0, 0, 0);
  auto pasteFilter = itk::PasteImageFilter<ProjectionStackType, ProjectionStackType, ProjectionStackType>::New();
  pasteFilter->SetDestinationImage(projectionsSource->GetOutput());

#ifdef USE_CUDA
  std::string signalFileName = "signal_4DConjugatedGradient_cuda.txt";
#else
  std::string signalFileName = "signal_4DConjugatedGradient.txt";
#endif

  std::ofstream signalFile(signalFileName.c_str());
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
  {
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
    e1->SetDensity(2.);
    e1->SetAxis(semiprincipalaxis);
    e1->SetCenter(center);
    e1->SetAngle(0.);
    e1->InPlaceOff();
    e1->Update();

    // Ellipse 2
    auto e2 = REIType::New();
    semiprincipalaxis.Fill(8.);
    center[0] = 4 * (itk::Math::abs((4 + noProj) % 8 - 4.) - 2.);
    center[1] = 0.;
    center[2] = 0.;
    e2->SetInput(e1->GetOutput());
    e2->SetGeometry(oneProjGeometry);
    e2->SetDensity(-1.);
    e2->SetAxis(semiprincipalaxis);
    e2->SetCenter(center);
    e2->SetAngle(0.);
    e2->Update();

    // Adding each projection to the projection stack
    if (noProj > 0) // After the first projection, we use the output as input
    {
      ProjectionStackType::Pointer wholeImage = pasteFilter->GetOutput();
      wholeImage->DisconnectPipeline();
      pasteFilter->SetDestinationImage(wholeImage);
    }
    pasteFilter->SetSourceImage(e2->GetOutput());
    pasteFilter->SetSourceRegion(e2->GetOutput()->GetLargestPossibleRegion());
    pasteFilter->SetDestinationIndex(destinationIndex);
    pasteFilter->UpdateLargestPossibleRegion();
    destinationIndex[2]++;

    // Signal
    signalFile << (noProj % 8) / 8. << std::endl;
  }
  signalFile.close();

  // Ground truth
  auto * Volumes = new VolumeType::Pointer[fourDSize[3]];
  auto   join = itk::JoinSeriesImageFilter<VolumeType, VolumeSeriesType>::New();

  for (itk::SizeValueType n = 0; n < fourDSize[3]; n++)
  {
    // Ellipse 1
    using DEType = rtk::DrawEllipsoidImageFilter<VolumeType, VolumeType>;
    auto de1 = DEType::New();
    de1->SetInput(tomographySource->GetOutput());
    de1->SetDensity(2.);
    DEType::VectorType axis;
    axis.Fill(60.);
    axis[1] = 30;
    de1->SetAxis(axis);
    DEType::VectorType center;
    center.Fill(0.);
    de1->SetCenter(center);
    de1->SetAngle(0.);
    de1->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION(de1->Update())

    // Ellipse 2
    auto de2 = DEType::New();
    de2->SetInput(de1->GetOutput());
    de2->SetDensity(-1.);
    DEType::VectorType axis2;
    axis2.Fill(8.);
    de2->SetAxis(axis2);
    DEType::VectorType center2;
    center2[0] = 4 * (itk::Math::abs((4 + n) % 8 - 4.) - 2.);
    center2[1] = 0.;
    center2[2] = 0.;
    de2->SetCenter(center2);
    de2->SetAngle(0.);
    de2->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION(de2->Update());

    Volumes[n] = de2->GetOutput();
    Volumes[n]->DisconnectPipeline();
    join->SetInput(n, Volumes[n]);
  }
  join->Update();

  // Read the phases file
  auto phaseReader = rtk::PhasesToInterpolationWeights::New();
  phaseReader->SetFileName(signalFileName);
  phaseReader->SetNumberOfReconstructedFrames(fourDSize[3]);
  phaseReader->Update();

  // Set the forward and back projection filters to be used
  using ConjugateGradientFilterType =
    rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>;
  auto conjugategradient = ConjugateGradientFilterType::New();
  conjugategradient->SetInputVolumeSeries(fourdSource->GetOutput());
  conjugategradient->SetInputProjectionStack(pasteFilter->GetOutput());
  conjugategradient->SetGeometry(geometry);
  conjugategradient->SetNumberOfIterations(3);
  conjugategradient->SetWeights(phaseReader->GetOutput());
  conjugategradient->SetSignal(rtk::ReadSignalFile(signalFileName));

  std::cout
    << "\n\n****** Case 1: Joseph forward projector, Voxel-Based back projector, CPU interpolation and splat ******"
    << std::endl;

  conjugategradient->SetBackProjectionFilter(ConjugateGradientFilterType::BP_VOXELBASED);
  conjugategradient->SetForwardProjectionFilter(ConjugateGradientFilterType::FP_JOSEPH);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(conjugategradient->Update());

  CheckImageQuality<VolumeSeriesType>(conjugategradient->GetOutput(), join->GetOutput(), 0.4, 12, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 2: CUDA ray cast forward projector, CUDA Voxel-Based back projector, GPU interpolation "
               "and splat ******"
            << std::endl;

  conjugategradient->SetBackProjectionFilter(ConjugateGradientFilterType::BP_CUDAVOXELBASED);
  conjugategradient->SetForwardProjectionFilter(ConjugateGradientFilterType::FP_CUDARAYCAST);
  conjugategradient->SetCudaConjugateGradient(true);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(conjugategradient->Update());

  CheckImageQuality<VolumeSeriesType>(conjugategradient->GetOutput(), join->GetOutput(), 0.4, 12, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  itksys::SystemTools::RemoveFile(signalFileName.c_str());
  delete[] Volumes;

  return EXIT_SUCCESS;
}
