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

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfProjectionImages = 3;
#else
  const unsigned int NumberOfProjectionImages = 128;
#endif

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  origin[0] = -63.;
  origin[1] = -31.;
  origin[2] = -63.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 32;
  size[1] = 32;
  size[2] = 32;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
#else
  size[0] = 64;
  size[1] = 32;
  size[2] = 64;
  spacing[0] = 2.;
  spacing[1] = 2.;
  spacing[2] = 2.;
#endif
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -254.;
  origin[1] = -254.;
  origin[2] = -254.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 32;
  size[1] = 32;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 32.;
  spacing[1] = 32.;
  spacing[2] = 32.;
#else
  size[0] = 128;
  size[1] = 128;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
#endif
  projectionsSource->SetOrigin( origin );
  projectionsSource->SetSpacing( spacing );
  projectionsSource->SetSize( size );
  projectionsSource->SetConstant( 0. );

  ConstantImageSourceType::Pointer oneProjectionSource = ConstantImageSourceType::New();
  size[2] = 1;
  oneProjectionSource->SetOrigin( origin );
  oneProjectionSource->SetSpacing( spacing );
  oneProjectionSource->SetSize( size );
  oneProjectionSource->SetConstant( 0. );

  // Geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();

  // Projections
  typedef rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType> REIType;
  typedef itk::PasteImageFilter <OutputImageType, OutputImageType, OutputImageType > PasteImageFilterType;
  OutputImageType::IndexType destinationIndex;
  destinationIndex[0] = 0;
  destinationIndex[1] = 0;
  destinationIndex[2] = 0;
  PasteImageFilterType::Pointer pasteFilter = PasteImageFilterType::New();

  std::ofstream signalFile("signal.txt");
  OutputImageType::Pointer wholeImage = projectionsSource->GetOutput();
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    {
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

    // Geometry object
    GeometryType::Pointer oneProjGeometry = GeometryType::New();
    oneProjGeometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

    // Ellipse 1
    REIType::Pointer e1 = REIType::New();
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
    center[0] = 4*(vcl_abs( (4+noProj) % 8 - 4.) - 2.);
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
    pasteFilter->Update();
    wholeImage = pasteFilter->GetOutput();
    destinationIndex[2]++;

    // Signal
    signalFile << (noProj % 8) / 8. << std::endl;
    }

  // Create vector field
  typedef itk::Vector<float,3>                                                 DVFPixelType;
  typedef itk::Image< DVFPixelType, 3 >                                        DVFImageType;
  typedef rtk::CyclicDeformationImageFilter< DVFImageType >                    DeformationType;
  typedef itk::ImageRegionIteratorWithIndex< DeformationType::InputImageType > IteratorType;

  DeformationType::InputImageType::Pointer deformationField;
  deformationField = DeformationType::InputImageType::New();

  DeformationType::InputImageType::IndexType startMotion;
  startMotion[0] = 0; // first index on X
  startMotion[1] = 0; // first index on Y
  startMotion[2] = 0; // first index on Z
  startMotion[3] = 0; // first index on t
  DeformationType::InputImageType::SizeType sizeMotion;
  sizeMotion[0] = 64; // size along X
  sizeMotion[1] = 64; // size along Y
  sizeMotion[2] = 64; // size along Z
  sizeMotion[3] = 2;  // size along t
  DeformationType::InputImageType::PointType originMotion;
  originMotion[0] = (sizeMotion[0]-1)*(-0.5); // size along X
  originMotion[1] = (sizeMotion[1]-1)*(-0.5); // size along Y
  originMotion[2] = (sizeMotion[2]-1)*(-0.5); // size along Z
  originMotion[3] = 0.;
  DeformationType::InputImageType::RegionType regionMotion;
  regionMotion.SetSize( sizeMotion );
  regionMotion.SetIndex( startMotion );
  deformationField->SetRegions( regionMotion );
  deformationField->SetOrigin(originMotion);
  deformationField->Allocate();

  // Vector Field initilization
  DVFPixelType vec;
  vec.Fill(0.);
  IteratorType inputIt( deformationField, deformationField->GetLargestPossibleRegion() );
  for ( inputIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt)
    {
    if(inputIt.GetIndex()[3]==0)
      vec[0] = -8.;
    else
      vec[0] = 8.;
    inputIt.Set(vec);
    }

  // Create cyclic deformation
  DeformationType::Pointer def = DeformationType::New();
  def->SetInput(deformationField);
  typedef rtk::FDKWarpBackProjectionImageFilter<OutputImageType, OutputImageType, DeformationType> WarpBPType;
  WarpBPType::Pointer bp = WarpBPType::New();
  bp->SetDeformation(def);
  bp->SetGeometry( geometry.GetPointer() );

  // FDK reconstruction filtering
#ifdef USE_CUDA
  typedef rtk::CudaFDKConeBeamReconstructionFilter                FDKType;
#else
  typedef rtk::FDKConeBeamReconstructionFilter< OutputImageType > FDKType;
#endif
  FDKType::Pointer feldkamp = FDKType::New();
  feldkamp->SetInput( 0, tomographySource->GetOutput() );
  feldkamp->SetInput( 1, wholeImage );
  feldkamp->SetGeometry( geometry );
  def->SetSignalFilename("signal.txt");
  feldkamp.GetPointer()->SetBackProjectionFilter( bp.GetPointer() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );

  // FOV
  typedef rtk::FieldOfViewImageFilter<OutputImageType, OutputImageType> FOVFilterType;
  FOVFilterType::Pointer fov=FOVFilterType::New();
  fov->SetInput(0, feldkamp->GetOutput());
  fov->SetProjectionsStack( wholeImage.GetPointer() );
  fov->SetGeometry( geometry );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fov->Update() );

  // Create a reference object (in this case a 3D phantom reference).
  // Ellipse 1
  typedef rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType> DEType;
  DEType::Pointer e1 = DEType::New();
  e1->SetInput( tomographySource->GetOutput() );
  e1->SetDensity(2.);
  DEType::VectorType axis;
  axis.Fill(60.);
  e1->SetAxis(axis);
  DEType::VectorType center;
  center.Fill(0.);
  e1->SetCenter(center);
  e1->SetAngle(0.);
  e1->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( e1->Update() )

  // Ellipse 2
  DEType::Pointer e2 = DEType::New();
  e2->SetInput(e1->GetOutput());
  e2->SetDensity(-1.);
  DEType::VectorType axis2;
  axis2.Fill(8.);
  e2->SetAxis(axis2);
  DEType::VectorType center2;
  center2.Fill(0.);
  e2->SetCenter(center2);
  e2->SetAngle(0.);
  e2->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( e2->Update() )

  CheckImageQuality<OutputImageType>(fov->GetOutput(), e2->GetOutput(), 0.05, 22, 2.0);

  std::cout << "Test PASSED! " << std::endl;

  itksys::SystemTools::RemoveFile("signal.txt");

  return EXIT_SUCCESS;
}
