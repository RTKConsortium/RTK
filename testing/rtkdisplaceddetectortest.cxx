#include <itkImageRegionConstIterator.h>

#include "rtkTest.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkConstantImageSource.h"
#ifdef USE_CUDA
#  include "rtkCudaDisplacedDetectorImageFilter.h"
#else
#  include "rtkDisplacedDetectorImageFilter.h"
#endif

/**
 * \file rtkdisplaceddetectortest.cxx
 *
 * \brief Functional test for classes performing FDK reconstructions with a
 * displaced detector/source
 *
 * This test generates the projections of a simulated Shepp-Logan phantom and
 * different sets of geometries with different displaced detectors and sources.
 * Images are then reconstructed from the generated projection images and
 * compared to the expected results which is analytically computed.
 *
 * \author Simon Rit and Marc Vila
 */

int main(int, char**)
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
#ifdef USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
#endif

#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfProjectionImages = 3;
#else
  const unsigned int NumberOfProjectionImages = 180;
#endif

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = 2;
  spacing[0] = 254.;
  spacing[1] = 254.;
  spacing[2] = 254.;
#else
  size[0] = 128;
  size[1] = 128;
  size[2] = 128;
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
  size[0] = 2;
  size[1] = 2;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 508.;
  spacing[1] = 508.;
  spacing[2] = 508.;
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

  // Shepp Logan projections filter
  typedef rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  SLPType::Pointer slp = SLPType::New();
  slp->SetInput( projectionsSource->GetOutput() );

  // Displaced detector weighting
#ifdef USE_CUDA
  typedef rtk::CudaDisplacedDetectorImageFilter DDFType;
#else
  typedef rtk::DisplacedDetectorImageFilter<OutputImageType> DDFType;
#endif
  DDFType::Pointer ddf = DDFType::New();
  ddf->SetInput( slp->GetOutput() );

  // Create a reference object (in this case a 3D phantom reference).
  typedef rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType> DSLType;
  DSLType::Pointer dsl = DSLType::New();
  dsl->SetInput( tomographySource->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() );

  // FDK reconstruction filtering
  typedef rtk::FDKConeBeamReconstructionFilter<OutputImageType> FDKCPUType;
  FDKCPUType::Pointer feldkamp = FDKCPUType::New();
  feldkamp->SetInput( 0, tomographySource->GetOutput() );
  feldkamp->SetInput( 1, ddf->GetOutput() );

  std::cout << "\n\n****** Case 1: positive offset in geometry ******" << std::endl;
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  slp->SetGeometry(geometry);
  ddf->SetGeometry( geometry );
  feldkamp->SetGeometry( geometry );
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages, 120., 0.);
  slp->Update();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.061, 24, 2.0);

  std::cout << "\n\n****** Case 2: negative offset in geometry ******" << std::endl;
  geometry = GeometryType::New();
  slp->SetGeometry(geometry);
  ddf->SetGeometry( geometry );
  feldkamp->SetGeometry( geometry );
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages, -120., 0.);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.061, 24, 2.0);

  std::cout << "\n\n****** Case 3: no displacement ******" << std::endl;
  geometry = GeometryType::New();
  slp->SetGeometry(geometry);
  ddf->SetGeometry( geometry );
  feldkamp->SetGeometry( geometry );
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages);
  projectionsSource->SetOrigin(origin);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.061, 24, 2.0);

  std::cout << "\n\n****** Case 4: negative offset in origin ******" << std::endl;
  geometry = GeometryType::New();
  slp->SetGeometry(geometry);
  ddf->SetGeometry( geometry );
  feldkamp->SetGeometry( geometry );
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages);
  origin[0] = -400;
  projectionsSource->SetOrigin(origin);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.061, 24, 2.0);

  std::cout << "\n\n****** Case 5: positive offset in origin ******" << std::endl;
  origin[0] = -100;
  projectionsSource->SetOrigin(origin);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );
  CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.061, 24, 2.0);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
