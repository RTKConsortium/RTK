#include <itkImageRegionConstIterator.h>
#include <itkPasteImageFilter.h>
#include <itksys/SystemTools.hxx>

#include "rtkTestConfiguration.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkFDKWarpBackProjectionImageFilter.h"
#include "rtkCyclicDeformationImageFilter.h"

template<class TImage>
#if FAST_TESTS_NO_CHECKS
void CheckImageQuality(typename TImage::Pointer itkNotUsed(recon), typename TImage::Pointer itkNotUsed(ref))
{
}
#else
void CheckImageQuality(typename TImage::Pointer recon, typename TImage::Pointer ref)
{
  typedef itk::ImageRegionConstIterator<TImage> ImageIteratorType;
  ImageIteratorType itTest( recon, recon->GetBufferedRegion() );
  ImageIteratorType itRef( ref, ref->GetBufferedRegion() );

  typedef double ErrorType;
  ErrorType TestError = 0.;
  ErrorType EnerError = 0.;

  itTest.GoToBegin();
  itRef.GoToBegin();

  while( !itRef.IsAtEnd() )
    {
    typename TImage::PixelType TestVal = itTest.Get();
    typename TImage::PixelType RefVal = itRef.Get();
    TestError += vcl_abs(RefVal - TestVal);
    EnerError += vcl_pow(ErrorType(RefVal - TestVal), 2.);
    ++itTest;
    ++itRef;
    }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError/ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError/ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "MSE = " << MSE << std::endl;
  // PSNR
  ErrorType PSNR = 20*log10(2.0) - 10*log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (2.0-ErrorPerPixel)/2.0;
  std::cout << "QI = " << QI << std::endl;

  // Checking results
  if (ErrorPerPixel > 0.05)
  {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 0.05." << std::endl;
    exit( EXIT_FAILURE);
  }
  if (PSNR < 22.)
  {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 23" << std::endl;
    exit( EXIT_FAILURE);
  }
}
#endif

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
    e1->SetInput(oneProjectionSource->GetOutput());
    e1->SetGeometry(oneProjGeometry);
    e1->SetMultiplicativeConstant(2.);
    e1->SetSemiPrincipalAxisX(60.);
    e1->SetSemiPrincipalAxisY(60.);
    e1->SetSemiPrincipalAxisZ(60.);
    e1->SetCenterX(0.);
    e1->SetCenterY(0.);
    e1->SetCenterZ(0.);
    e1->SetRotationAngle(0.);
    e1->InPlaceOff();
    e1->Update();

    // Ellipse 2
    REIType::Pointer e2 = REIType::New();
    e2->SetInput(e1->GetOutput());
    e2->SetGeometry(oneProjGeometry);
    e2->SetMultiplicativeConstant(-1.);
    e2->SetSemiPrincipalAxisX(8.);
    e2->SetSemiPrincipalAxisY(8.);
    e2->SetSemiPrincipalAxisZ(8.);
    e2->SetCenterX( 4*(vcl_abs( (4+noProj) % 8 - 4.) - 2.) );
    e2->SetCenterY(0.);
    e2->SetCenterZ(0.);
    e2->SetRotationAngle(0.);
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
#elif USE_OPENCL
  typedef rtk::OpenCLFDKConeBeamReconstructionFilter              FDKType;
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
  e1->SetAttenuation(2.);
  DEType::VectorType axis(3, 60.);
  e1->SetAxis(axis);
  DEType::VectorType center(3, 0.);
  e1->SetCenter(center);
  e1->SetAngle(0.);
  e1->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( e1->Update() )

  // Ellipse 2
  DEType::Pointer e2 = DEType::New();
  e2->SetInput(e1->GetOutput());
  e2->SetAttenuation(-1.);
  DEType::VectorType axis2(3, 8.);
  e2->SetAxis(axis2);
  DEType::VectorType center2(3, 0.);
  e2->SetCenter(center2);
  e2->SetAngle(0.);
  e2->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( e2->Update() )

  CheckImageQuality<OutputImageType>(fov->GetOutput(), e2->GetOutput());

  std::cout << "Test PASSED! " << std::endl;

  itksys::SystemTools::RemoveFile("signal.txt");

  return EXIT_SUCCESS;
}
