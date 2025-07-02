#include <itkPasteImageFilter.h>
#include <itksys/SystemTools.hxx>

#include "rtkTest.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkFDKWarpBackProjectionImageFilter.h"
#include "rtkCyclicDeformationImageFilter.h"

/**
 * \file rtkmotioncompensatedfdktest.cxx
 *
 * \brief Functional tests for classes performing a motion compensated FDK
 * reconstruction.
 *
 * This test generates the projections of a phantom, which consists of two
 * ellipsoids (one of them moving). The resulting moving phantom is
 * reconstructed using motion compensation techniques and these generated
 * results are compared to the expected results (analytical computation).
 *
 * \author Simon Rit and Marc Vila
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputImageType = itk::Image<float, Dimension>;
#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 128;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto tomographySource = ConstantImageSourceType::New();
  auto origin = itk::MakePoint(-63., -31., -63.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(32, 32, 32);
  auto spacing = itk::MakeVector(8., 8., 8.);
#else
  auto size = itk::MakeSize(64, 32, 64);
  auto spacing = itk::MakeVector(2., 2., 2.);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  auto projectionsSource = ConstantImageSourceType::New();
  origin = itk::MakePoint(-254., -254., -254.);
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(32, 32, NumberOfProjectionImages);
  spacing = itk::MakeVector(32., 32., 32.);
#else
  size = itk::MakeSize(128, 128, NumberOfProjectionImages);
  spacing = itk::MakeVector(4., 4., 4.);
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
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType>;
  auto destinationIndex = itk::MakeIndex(0, 0, 0);
  auto pasteFilter = itk::PasteImageFilter<OutputImageType, OutputImageType, OutputImageType>::New();

  std::ofstream            signalFile("signal.txt");
  OutputImageType::Pointer wholeImage = projectionsSource->GetOutput();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
  {
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

    // Geometry object
    auto oneProjGeometry = GeometryType::New();
    oneProjGeometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

    // Ellipse 1
    auto e1 = REIType::New();
    auto semiprincipalaxis = itk::MakeVector(60., 60., 60.);
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
    e2->SetInput(e1->GetOutput());
    e2->SetGeometry(oneProjGeometry);
    e2->SetDensity(-1.);
    e2->SetAxis(itk::MakeVector(8., 8., 8.));
    e2->SetCenter(itk::MakeVector(4 * (itk::Math::abs((4 + noProj) % 8 - 4.) - 2.), 0., 0.));
    e2->SetAngle(0.);
    e2->Update();

    // Adding each projection to volume
    pasteFilter->SetSourceImage(e2->GetOutput());
    pasteFilter->SetDestinationImage(wholeImage);
    pasteFilter->SetSourceRegion(e2->GetOutput()->GetLargestPossibleRegion());
    pasteFilter->SetDestinationIndex(destinationIndex);
    pasteFilter->Update();
    wholeImage = pasteFilter->GetOutput();
    destinationIndex[2]++;

    // Signal
    signalFile << (noProj % 8) / 8. << std::endl;
  }
  signalFile.close();

  // Create vector field
  using DVFPixelType = itk::Vector<float, 3>;
  using DeformationType = rtk::CyclicDeformationImageFilter<itk::Image<DVFPixelType, 4>, itk::Image<DVFPixelType, 3>>;

  DeformationType::InputImageType::Pointer deformationField;
  deformationField = DeformationType::InputImageType::New();

  auto                                       sizeMotion = itk::MakeSize(64, 64, 64, 2);
  DeformationType::InputImageType::PointType originMotion;
  originMotion[0] = (sizeMotion[0] - 1) * (-0.5); // size along X
  originMotion[1] = (sizeMotion[1] - 1) * (-0.5); // size along Y
  originMotion[2] = (sizeMotion[2] - 1) * (-0.5); // size along Z
  originMotion[3] = 0.;
  DeformationType::InputImageType::RegionType regionMotion;
  regionMotion.SetSize(sizeMotion);
  deformationField->SetRegions(regionMotion);
  deformationField->SetOrigin(originMotion);
  deformationField->Allocate();

  // Vector Field initilization
  DVFPixelType vec;
  vec.Fill(0.);
  itk::ImageRegionIteratorWithIndex<DeformationType::InputImageType> inputIt(
    deformationField, deformationField->GetLargestPossibleRegion());
  for (inputIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt)
  {
    if (inputIt.GetIndex()[3] == 0)
      vec[0] = -8.;
    else
      vec[0] = 8.;
    inputIt.Set(vec);
  }

  // Create cyclic deformation
  auto def = DeformationType::New();
  def->SetInput(deformationField);
  auto bp = rtk::FDKWarpBackProjectionImageFilter<OutputImageType, OutputImageType, DeformationType>::New();
  bp->SetDeformation(def);
  bp->SetGeometry(geometry.GetPointer());

  // FDK reconstruction filtering
#ifdef USE_CUDA
  using FDKType = rtk::CudaFDKConeBeamReconstructionFilter;
#else
  using FDKType = rtk::FDKConeBeamReconstructionFilter<OutputImageType>;
#endif
  auto feldkamp = FDKType::New();
  feldkamp->SetInput(0, tomographySource->GetOutput());
  feldkamp->SetInput(1, wholeImage);
  feldkamp->SetGeometry(geometry);
  def->SetSignalFilename("signal.txt");
  feldkamp.GetPointer()->SetBackProjectionFilter(bp.GetPointer());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkamp->Update());

  // FOV
  auto fov = rtk::FieldOfViewImageFilter<OutputImageType, OutputImageType>::New();
  fov->SetInput(0, feldkamp->GetOutput());
  fov->SetProjectionsStack(wholeImage.GetPointer());
  fov->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(fov->Update());

  // Create a reference object (in this case a 3D phantom reference).
  // Ellipse 1
  using DEType = rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType>;
  auto e1 = DEType::New();
  e1->SetInput(tomographySource->GetOutput());
  e1->SetDensity(2.);
  e1->SetAxis(itk::MakeVector(60., 60., 60.));
  e1->SetCenter(itk::MakeVector(0., 0., 0.));
  e1->SetAngle(0.);
  e1->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(e1->Update())

  // Ellipse 2
  auto e2 = DEType::New();
  e2->SetInput(e1->GetOutput());
  e2->SetDensity(-1.);
  e2->SetAxis(itk::MakeVector(8., 8., 8.));
  e2->SetCenter(itk::MakeVector(0., 0., 0.));
  e2->SetAngle(0.);
  e2->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(e2->Update())

  CheckImageQuality<OutputImageType>(fov->GetOutput(), e2->GetOutput(), 0.05, 22, 2.0);

  std::cout << "Test PASSED! " << std::endl;

  itksys::SystemTools::RemoveFile("signal.txt");

  return EXIT_SUCCESS;
}
