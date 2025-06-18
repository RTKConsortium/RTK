#include <itkImageFileReader.h>
#include <itkPasteImageFilter.h>
#include <itkConfigure.h>

#include "rtkTest.h"
#include "rtkAmsterdamShroudImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConfiguration.h"
#include "rtkReg1DExtractShroudSignalImageFilter.h"
#include "rtkDPExtractShroudSignalImageFilter.h"
#include "rtkMacro.h"
#include "rtkExtractPhaseImageFilter.h"

/**
 * \file rtkamsterdamshroudtest.cxx
 *
 * \brief Functional test for classes performing Amsterdam Shroud and breathing signal extraction algorithms
 *
 * This test generates an Amsterdam Shroud image from a moving simulated phantom
 * and extracts the breathing signal using two different methods, reg1D and D
 * algorithms. The generated results are compared to the expected results,
 * read from a baseline image in the MetaIO file format and hard-coded,
 * respectively.
 *
 * \author Marc Vila
 */
int
main(int argc, char * argv[])
{
  constexpr unsigned int Dimension = 3;
  using reg1DPixelType = double;
  using OutputPixelType = float;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  using reg1DImageType = itk::Image<reg1DPixelType, Dimension - 2>;
#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 100;
#endif

  if (argc < 3)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " croppedRefObject refObject " << std::endl;
    return EXIT_FAILURE;
  }

  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometryMain = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometryMain->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);

  // Create a stack of empty projection images
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;

  // Adjust size according to geometry and for just one projection
  ConstantImageSourceType::Pointer constantImageSourceSingleProjection = ConstantImageSourceType::New();
  auto                             origin = itk::MakePoint(-50., -50., -158.75);
#if FAST_TESTS_NO_CHECKS
  auto spacing = itk::MakeVector(106., 106., 2.5);
  auto size = itk::MakeSize(4, 4, 1);
#else
  auto spacing = itk::MakeVector(2.5, 2.5, 2.5);
  auto size = itk::MakeSize(128, 128, 1);
#endif
  constantImageSourceSingleProjection->SetOrigin(origin);
  constantImageSourceSingleProjection->SetSpacing(spacing);
  constantImageSourceSingleProjection->SetSize(size);
  constantImageSourceSingleProjection->SetConstant(0.);

  // Adjust size according to geometry
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  size[2] = NumberOfProjectionImages;
  constantImageSource->SetOrigin(origin);
  constantImageSource->SetSpacing(spacing);
  constantImageSource->SetSize(size);
  constantImageSource->SetConstant(0.);

  using PasteImageFilterType = itk::PasteImageFilter<OutputImageType, OutputImageType, OutputImageType>;
  OutputImageType::IndexType destinationIndex;
  destinationIndex[0] = 0;
  destinationIndex[1] = 0;
  destinationIndex[2] = 0;

  PasteImageFilterType::Pointer pasteFilter = PasteImageFilterType::New();

  // Single projection
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType>;
  double                 reiSize = 80.;
  double                 sinus = 0.;
  constexpr unsigned int Cycles = 4;

  OutputImageType::Pointer wholeImage = constantImageSource->GetOutput();
  GeometryType::Pointer    geometryFull = GeometryType::New();
  for (unsigned int i = 1; i <= size[2]; i++)
  {
    // Geometry object
    GeometryType::Pointer geometry = GeometryType::New();
    geometry->AddProjection(1200., 1500., i * 360 / size[2]);
    geometryFull->AddProjection(1200., 1500., i * 360 / size[2]);

    // Ellipse 1
    REIType::Pointer e1 = REIType::New();
    e1->SetInput(constantImageSourceSingleProjection->GetOutput());
    e1->SetGeometry(geometry);
    e1->SetDensity(2.);
    e1->SetAxis(itk::MakeVector(88.32, 115.2, 117.76));
    e1->SetCenter(itk::MakeVector(0., 0., 0.));
    e1->SetAngle(0.);
    e1->InPlaceOff();
    e1->Update();

    // Ellipse 2
    REIType::Pointer e2 = REIType::New();
    e2->SetInput(e1->GetOutput());
    e2->SetGeometry(geometry);
    e2->SetDensity(-1.98);
    e2->SetAxis(itk::MakeVector(35., reiSize - sinus, reiSize - sinus));
    e2->SetCenter(itk::MakeVector(-37., 0., 0.));
    e2->SetAngle(0.);
    e2->Update();

    // Ellipse 3
    REIType::Pointer e3 = REIType::New();
    e3->SetInput(e2->GetOutput());
    e3->SetGeometry(geometry);
    e3->SetDensity(-1.98);
    e3->SetAxis(itk::MakeVector(35., reiSize - sinus, reiSize - sinus));
    e3->SetCenter(itk::MakeVector(37., 0., 0.));
    e3->SetAngle(0.);
    e3->Update();

    // Ellipse 4
    REIType::Pointer e4 = REIType::New();
    e4->SetInput(e3->GetOutput());
    e4->SetGeometry(geometry);
    e4->SetDensity(1.42);
    e4->SetAxis(itk::MakeVector(8., 8., 8.));
    e4->SetCenter(itk::MakeVector(-40., 0., 0.));
    e4->SetAngle(0.);

    // Creating movement
    sinus = 15 * sin(i * 2 * itk::Math::pi / (size[2] / Cycles));

    // Generating projection
    e4->Update();

    // Adding each projection to volume
    pasteFilter->SetSourceImage(e4->GetOutput());
    pasteFilter->SetDestinationImage(wholeImage);
    pasteFilter->SetSourceRegion(e4->GetOutput()->GetLargestPossibleRegion());
    pasteFilter->SetDestinationIndex(destinationIndex);
    pasteFilter->Update();
    wholeImage = pasteFilter->GetOutput();
    destinationIndex[2]++;
  }

  std::cout << "\n\n****** Case 0: Amsterdam Shroud Image with crop ******" << std::endl;

  // Amsterdam shroud
  using ShroudFilterType = rtk::AmsterdamShroudImageFilter<OutputImageType>;
  ShroudFilterType::Pointer shroudFilter = ShroudFilterType::New();
  shroudFilter->SetInput(pasteFilter->GetOutput());
  shroudFilter->SetGeometry(geometryFull);
  ShroudFilterType::PointType center(0.), offset1(-40), offset2(40);
  center[0] = 37.;
  center[1] = 80.;
  for (int i = 0; i < 3; i++)
  {
    offset1[i] += center[i];
    offset2[i] += center[i];
  }
  shroudFilter->SetCorner1(offset1);
  shroudFilter->SetCorner2(offset2);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(shroudFilter->Update());

  // Read reference object
  using ReaderAmsterdamType = itk::ImageFileReader<ShroudFilterType::OutputImageType>;
  ReaderAmsterdamType::Pointer reader2 = ReaderAmsterdamType::New();
  reader2->SetFileName(argv[1]);
  reader2->Update();

  CheckImageQuality<ShroudFilterType::OutputImageType>(
    shroudFilter->GetOutput(), reader2->GetOutput(), 1.20e-6, 185, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  std::cout << "\n\n****** Case 1: Amsterdam Shroud Image without crop ******" << std::endl;

  // Amsterdam shroud
  shroudFilter->SetGeometry(nullptr);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(shroudFilter->Update());

  // Read reference object
  reader2->SetFileName(argv[2]);
  reader2->Update();

  CheckImageQuality<ShroudFilterType::OutputImageType>(
    shroudFilter->GetOutput(), reader2->GetOutput(), 1.20e-6, 185, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Breathing signal calculated by reg1D algorithm ******\n" << std::endl;

  // Estimation of breathing signal
  using reg1DFilterType = rtk::Reg1DExtractShroudSignalImageFilter<reg1DPixelType, reg1DPixelType>;
  reg1DImageType::Pointer  reg1DSignal;
  reg1DFilterType::Pointer reg1DFilter = reg1DFilterType::New();
  reg1DFilter->SetInput(reader2->GetOutput());
  reg1DFilter->Update();
  reg1DSignal = reg1DFilter->GetOutput();

  // Test Reference
  float reg1D[100] = {
    0.f,     4.5f,    8.625f,  12.25f, 15.f,     16.875f,  17.625f,  17.375f,  16.125f,  13.875f,  10.75f,   7.125f,
    3.f,     -1.25f,  -5.375f, -9.f,   -12.125f, -14.25f,  -15.625f, -16.125f, -15.5f,   -13.75f,  -11.f,    -7.5f,
    -3.375f, 1.125f,  5.5f,    9.5f,   13.f,     15.875f,  17.75f,   18.625f,  18.375f,  17.25f,   15.f,     11.875f,
    8.125f,  3.875f,  -0.625f, -5.f,   -8.875f,  -12.125f, -14.25f,  -15.375f, -15.375f, -14.625f, -12.875f, -10.25f,
    -6.875f, -2.75f,  1.625f,  6.f,    10.125f,  13.625f,  16.375f,  18.25f,   19.f,     18.75f,   17.5f,    15.25f,
    12.125f, 8.5f,    4.375f,  0.125f, -4.f,     -7.625f,  -10.75f,  -12.875f, -14.25f,  -14.75f,  -14.125f, -12.375f,
    -9.625f, -6.125f, -2.f,    2.5f,   6.875f,   10.875f,  14.375f,  17.25f,   19.125f,  20.f,     19.75f,   18.625f,
    16.375f, 13.125f, 9.375f,  5.125f, 0.625f,   -3.75f,   -7.625f,  -10.875f, -13.f,    -14.125f, -14.125f, -13.375f,
    -11.5f,  -8.875f, -5.375f, -1.25f
  };

  // Checking for possible errors
  float                                         zeroValue = 1e-12;
  float                                         sum = 0.;
  unsigned int                                  i = 0;
  itk::ImageRegionConstIterator<reg1DImageType> it(reg1DSignal, reg1DSignal->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it, i++)
    sum += itk::Math::abs(reg1D[i] - it.Get());

  if (sum <= zeroValue)
    std::cout << "Test PASSED! " << std::endl;
  else
  {
    std::cerr << "Test FAILED! "
              << "Breathing signal does not match, absolute difference " << sum << " instead of 0." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "\n\n****** Case 3: Breathing signal calculated by DP algorithm ******\n" << std::endl;

  // Estimation of breathing signal
  using DPFilterType = rtk::DPExtractShroudSignalImageFilter<reg1DPixelType, reg1DPixelType>;
  reg1DImageType::Pointer DPSignal;
  DPFilterType::Pointer   DPFilter = DPFilterType::New();
  DPFilter->SetInput(reader2->GetOutput());
  DPFilter->SetAmplitude(20.);
  DPFilter->Update();
  DPSignal = DPFilter->GetOutput();

  // Test Reference
  float DP[100] = { 2.5f,  7.5f,   12.5f,  15.f,   17.5f,  20.f,   20.f,   20.f,   20.f,   17.5f, 12.5f, 10.f,  5.f,
                    0.f,   -5.f,   -7.5f,  -10.f,  -12.5f, -15.f,  -15.f,  -15.f,  -12.5f, -10.f, -7.5f, -2.5f, 2.5f,
                    7.5f,  10.f,   15.f,   17.5f,  20.f,   20.f,   20.f,   17.5f,  15.f,   12.5f, 10.f,  5.f,   0.f,
                    -5.f,  -7.5f,  -10.f,  -12.5f, -15.f,  -15.f,  -12.5f, -12.5f, -10.f,  -5.f,  0.f,   2.5f,  7.5f,
                    12.5f, 15.f,   17.5f,  20.f,   20.f,   20.f,   20.f,   17.5f,  12.5f,  10.f,  5.f,   0.f,   -5.f,
                    -7.5f, -10.f,  -12.5f, -15.f,  -15.f,  -15.f,  -12.5f, -10.f,  -7.5f,  -2.5f, 2.5f,  7.5f,  10.f,
                    15.f,  17.5f,  20.f,   20.f,   20.f,   17.5f,  15.f,   12.5f,  10.f,   5.f,   0.f,   -5.f,  -7.5f,
                    -10.f, -12.5f, -15.f,  -15.f,  -12.5f, -12.5f, -10.f,  -5.f,   0.f };

  // Checking for possible errors
  sum = 0.;
  i = 0;
  itk::ImageRegionConstIterator<reg1DImageType> itDP(DPSignal, DPSignal->GetLargestPossibleRegion());
  for (itDP.GoToBegin(); !itDP.IsAtEnd(); ++itDP, i++)
    sum += itk::Math::abs(DP[i] - itDP.Get());

  if (sum <= zeroValue)
    std::cout << "Test PASSED! " << std::endl;
  else
  {
    std::cerr << "Test FAILED! "
              << "Breathing signal does not match, absolute difference " << sum << " instead of 0." << std::endl;
    exit(EXIT_FAILURE);
  }

#if defined(USE_FFTWD)
  std::cout << "\n\n****** Extract phase from case 3 ******\n" << std::endl;

  // Check phase
  using PhaseType = rtk::ExtractPhaseImageFilter<reg1DImageType>;
  PhaseType::Pointer phaseFilt = PhaseType::New();
  phaseFilt->SetInput(DPSignal);
  phaseFilt->SetUnsharpMaskSize(53);
  std::cout << "Unsharp mask size is " << phaseFilt->GetUnsharpMaskSize() << std::endl;
  phaseFilt->SetMovingAverageSize(3);
  std::cout << "Moving average size is " << phaseFilt->GetMovingAverageSize() << std::endl;
  phaseFilt->SetModel(PhaseType::LOCAL_PHASE);
  phaseFilt->Update();
  reg1DImageType * phase = phaseFilt->GetOutput();

  // Checking for possible errors
  float refLocalPhase[100] = { 0.76f, 0.80f, 0.85f, 0.89f, 0.94f, 0.98f, 0.02f, 0.06f, 0.10f, 0.14f, 0.18f, 0.22f,
                               0.26f, 0.30f, 0.34f, 0.37f, 0.40f, 0.44f, 0.47f, 0.51f, 0.55f, 0.59f, 0.63f, 0.66f,
                               0.70f, 0.75f, 0.78f, 0.82f, 0.86f, 0.90f, 0.95f, 0.99f, 0.03f, 0.07f, 0.11f, 0.14f,
                               0.18f, 0.23f, 0.28f, 0.32f, 0.36f, 0.40f, 0.43f, 0.48f, 0.52f, 0.56f, 0.60f, 0.64f,
                               0.68f, 0.72f, 0.76f, 0.79f, 0.83f, 0.87f, 0.91f, 0.95f, 0.98f, 0.02f, 0.06f, 0.11f,
                               0.15f, 0.19f, 0.23f, 0.27f, 0.32f, 0.36f, 0.39f, 0.43f, 0.47f, 0.51f, 0.55f, 0.59f,
                               0.63f, 0.67f, 0.71f, 0.75f, 0.79f, 0.82f, 0.86f, 0.90f, 0.94f, 0.98f, 0.01f, 0.05f,
                               0.08f, 0.12f, 0.15f, 0.19f, 0.24f, 0.28f, 0.32f, 0.35f, 0.39f, 0.44f, 0.49f, 0.53f,
                               0.58f, 0.63f, 0.68f, 0.72f };
  itk::ImageRegionConstIterator<reg1DImageType> itPhaseLocal(phase, phase->GetLargestPossibleRegion());
  for (sum = 0, i = 0; !itPhaseLocal.IsAtEnd(); ++itPhaseLocal, i++)
    sum += itk::Math::abs(refLocalPhase[i] - itPhaseLocal.Get());
  std::cout << "LOCAL_PHASE... ";
  if (sum <= 0.27)
    std::cout << "Test PASSED! " << std::endl;
  else
  {
    std::cerr << "Test FAILED! Local phase does not match ref, absolute difference " << sum << " instead of 0."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  phaseFilt->SetModel(PhaseType::LINEAR_BETWEEN_MAXIMA);
  phaseFilt->Update();

  float refMaxPhase[100] = { 0.77f, 0.81f, 0.85f, 0.88f, 0.92f, 0.96f, 0.00f, 0.04f, 0.08f, 0.12f, 0.15f, 0.19f, 0.23f,
                             0.27f, 0.31f, 0.35f, 0.38f, 0.42f, 0.46f, 0.50f, 0.54f, 0.58f, 0.62f, 0.65f, 0.69f, 0.73f,
                             0.77f, 0.81f, 0.85f, 0.88f, 0.92f, 0.96f, 0.00f, 0.04f, 0.08f, 0.12f, 0.16f, 0.20f, 0.24f,
                             0.28f, 0.32f, 0.36f, 0.40f, 0.44f, 0.48f, 0.52f, 0.56f, 0.60f, 0.64f, 0.68f, 0.72f, 0.76f,
                             0.80f, 0.84f, 0.88f, 0.92f, 0.96f, 0.00f, 0.04f, 0.08f, 0.12f, 0.16f, 0.20f, 0.24f, 0.28f,
                             0.32f, 0.36f, 0.40f, 0.44f, 0.48f, 0.52f, 0.56f, 0.60f, 0.64f, 0.68f, 0.72f, 0.76f, 0.80f,
                             0.84f, 0.88f, 0.92f, 0.96f, 0.00f, 0.04f, 0.08f, 0.12f, 0.16f, 0.20f, 0.24f, 0.28f, 0.32f,
                             0.36f, 0.40f, 0.44f, 0.48f, 0.52f, 0.56f, 0.60f, 0.64f, 0.68f };

  itk::ImageRegionConstIterator<reg1DImageType> itPhaseMax(phase, phase->GetLargestPossibleRegion());
  for (sum = 0, i = 0; !itPhaseMax.IsAtEnd(); ++itPhaseMax, i++)
    sum += itk::Math::abs(refMaxPhase[i] - itPhaseMax.Get());
  std::cout << "LINEAR_BETWEEN_MAXIMA... ";
  if (sum <= 0.081)
    std::cout << "Test PASSED! " << std::endl;
  else
  {
    std::cerr << "Test FAILED! Linear phase between max does not match ref, absolute difference " << sum
              << " instead of 0." << std::endl;
    exit(EXIT_FAILURE);
  }

  phaseFilt->SetModel(PhaseType::LINEAR_BETWEEN_MINIMA);
  phaseFilt->Update();

  float refMinPhase[100] = { 0.24f, 0.28f, 0.32f, 0.36f, 0.40f, 0.44f, 0.48f, 0.52f, 0.56f, 0.60f, 0.64f, 0.68f, 0.72f,
                             0.76f, 0.80f, 0.84f, 0.88f, 0.92f, 0.96f, 0.00f, 0.04f, 0.08f, 0.12f, 0.16f, 0.20f, 0.24f,
                             0.28f, 0.32f, 0.36f, 0.40f, 0.44f, 0.48f, 0.52f, 0.56f, 0.60f, 0.64f, 0.68f, 0.72f, 0.76f,
                             0.80f, 0.84f, 0.88f, 0.92f, 0.96f, 0.00f, 0.04f, 0.08f, 0.12f, 0.16f, 0.20f, 0.24f, 0.28f,
                             0.32f, 0.36f, 0.40f, 0.44f, 0.48f, 0.52f, 0.56f, 0.60f, 0.64f, 0.68f, 0.72f, 0.76f, 0.80f,
                             0.84f, 0.88f, 0.92f, 0.96f, 0.00f, 0.04f, 0.08f, 0.12f, 0.17f, 0.21f, 0.25f, 0.29f, 0.33f,
                             0.38f, 0.42f, 0.46f, 0.50f, 0.54f, 0.58f, 0.62f, 0.67f, 0.71f, 0.75f, 0.79f, 0.83f, 0.88f,
                             0.92f, 0.96f, 0.00f, 0.04f, 0.08f, 0.12f, 0.17f, 0.21f, 0.25f };

  itk::ImageRegionConstIterator<reg1DImageType> itPhaseMin(phase, phase->GetLargestPossibleRegion());
  for (sum = 0, i = 0; !itPhaseMin.IsAtEnd(); ++itPhaseMin, i++)
    sum += itk::Math::abs(refMinPhase[i] - itPhaseMin.Get());
  std::cout << "LINEAR_BETWEEN_MINIMA... ";
  if (sum <= 0.076)
    std::cout << "Test PASSED! " << std::endl;
  else
  {
    std::cerr << "Test FAILED! Linear phase between min does not match ref, absolute difference " << sum
              << " instead of 0." << std::endl;
    exit(EXIT_FAILURE);
  }
#endif

  return EXIT_SUCCESS;
}
