#include <itksys/SystemTools.hxx>

#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkSelectOneProjectionPerCycleImageFilter.h"

/**
 * \file rtkmotioncompensatedfdktest.cxx
 *
 * \brief Check that the correct projection is selected in 3 cycles
 *
 * The test generates projections of 3 cycles and the corresponding signal
 * and check that rtk::SelectOneProjectionPerCycleImageFilter does a proper job.
 *
 * \author Simon Rit
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  constexpr unsigned int NumberOfProjectionImages = 24;

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::PointType   origin;
  ConstantImageSourceType::SizeType    size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  ConstantImageSourceType::Pointer projectionsSourceRef = ConstantImageSourceType::New();
  origin[0] = -254.;
  origin[1] = -254.;
  origin[2] = -254.;
  size[0] = 128;
  size[1] = 128;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);
  size[2] = 3;
  projectionsSourceRef->SetOrigin(origin);
  projectionsSourceRef->SetSpacing(spacing);
  projectionsSourceRef->SetSize(size);
  projectionsSourceRef->SetConstant(0.);

  ConstantImageSourceType::Pointer oneProjectionSource = ConstantImageSourceType::New();
  size[2] = 1;
  oneProjectionSource->SetOrigin(origin);
  oneProjectionSource->SetSpacing(spacing);
  oneProjectionSource->SetSize(size);
  oneProjectionSource->SetConstant(0.);

  // Geometry objects
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  GeometryType::Pointer geometryRef = GeometryType::New();

  // Projections
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType>;
  using PasteImageFilterType = itk::PasteImageFilter<OutputImageType, OutputImageType, OutputImageType>;
  OutputImageType::IndexType destinationIndex, destinationIndexRef;
  destinationIndex.Fill(0);
  destinationIndexRef.Fill(0);
  PasteImageFilterType::Pointer pasteFilter = PasteImageFilterType::New();

  std::string              signalFileName = "signal_SelectOneProjPerCycle.txt";
  std::ofstream            signalFile(signalFileName.c_str());
  OutputImageType::Pointer wholeImage = projectionsSource->GetOutput();
  OutputImageType::Pointer wholeImageRef = projectionsSourceRef->GetOutput();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
  {
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);
    if (noProj % 8 == 3)
      geometryRef->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

    // Geometry object
    GeometryType::Pointer oneProjGeometry = GeometryType::New();
    oneProjGeometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

    // Ellipse 1
    REIType::Pointer    e1 = REIType::New();
    REIType::VectorType semiprincipalaxis, center;
    semiprincipalaxis.Fill(60.);
    center.Fill(0.);
    e1->SetInput(oneProjectionSource->GetOutput());
    e1->SetGeometry(oneProjGeometry);
    e1->SetDensity(2.);
    e1->SetAxis(semiprincipalaxis);
    e1->SetCenter(center);
    e1->SetAngle(0.);
    e1->InPlaceOff();
    e1->Update();

    // Ellipse 2
    REIType::Pointer e2 = REIType::New();
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

    // Adding each projection to volume
    pasteFilter->SetSourceImage(e2->GetOutput());
    pasteFilter->SetDestinationImage(wholeImage);
    pasteFilter->SetSourceRegion(e2->GetOutput()->GetLargestPossibleRegion());
    pasteFilter->SetDestinationIndex(destinationIndex);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(pasteFilter->UpdateLargestPossibleRegion());
    wholeImage = pasteFilter->GetOutput();
    wholeImage->DisconnectPipeline();
    destinationIndex[2]++;

    if (noProj % 8 == 3)
    {
      // Adding each projection to volume ref
      pasteFilter->SetSourceImage(e2->GetOutput());
      pasteFilter->SetDestinationImage(wholeImageRef);
      pasteFilter->SetSourceRegion(e2->GetOutput()->GetLargestPossibleRegion());
      pasteFilter->SetDestinationIndex(destinationIndexRef);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(pasteFilter->UpdateLargestPossibleRegion(););
      wholeImageRef = pasteFilter->GetOutput();
      wholeImageRef->DisconnectPipeline();
      destinationIndexRef[2]++;
    }

    // Signal
    signalFile << (noProj % 8) / 8. << std::endl;
  }
  signalFile.close();

  // Select
  using SelectionType = rtk::SelectOneProjectionPerCycleImageFilter<OutputImageType>;
  SelectionType::Pointer select = SelectionType::New();
  select->SetInput(wholeImage);
  select->SetInputGeometry(geometry);
  select->SetPhase(0.4);
  select->SetSignalFilename(signalFileName);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(select->Update());

  CheckImageQuality<OutputImageType>(select->GetOutput(), wholeImageRef, 1e-12, 1e20, 1 - 1e-12);
  CheckGeometries(select->GetOutputGeometry(), geometryRef);

  std::cout << "Test PASSED! " << std::endl;

  itksys::SystemTools::RemoveFile(signalFileName.c_str());

  return EXIT_SUCCESS;
}
