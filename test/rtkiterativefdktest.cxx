#include <itkImageRegionConstIterator.h>
#include <itkStreamingImageFilter.h>

#include "rtkConfiguration.h"
#include "rtkTestConfiguration.h"
#include "rtkTest.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"

#ifdef USE_CUDA
#  include "rtkCudaIterativeFDKConeBeamReconstructionFilter.h"
#else
#  include "rtkIterativeFDKConeBeamReconstructionFilter.h"
#endif

/**
 * \file rtkiterativefdktest.cxx
 *
 * \brief Functional test for iterative FDK reconstruction
 *
 * This test generates the projections of a simulated Shepp-Logan phantom.
 * A CT image is reconstructed from the set of generated projection images
 * using the iterative FDK algorithm and the reconstructed CT image is compared
 * to the expected results which is analytically computed.
 *
 * \author Simon Rit
 */

int main(int, char** )
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
  size[0] = 32;
  size[1] = 32;
  size[2] = 32;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
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

  // Geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

  // Shepp Logan projections filter
  typedef rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  SLPType::Pointer slp=SLPType::New();
  slp->SetInput( projectionsSource->GetOutput() );
  slp->SetGeometry(geometry);
  slp->SetPhantomScale(116);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( slp->Update() );

  // Create a reference object (in this case a 3D phantom reference).
  typedef rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType> DSLType;
  DSLType::Pointer dsl = DSLType::New();
  dsl->SetInput( tomographySource->GetOutput() );
  dsl->SetPhantomScale(116);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() )

  // FDK reconstruction filtering
#ifdef USE_CUDA
  typedef rtk::CudaIterativeFDKConeBeamReconstructionFilter                FDKType;
#else
  typedef rtk::IterativeFDKConeBeamReconstructionFilter< OutputImageType > FDKType;
#endif
  FDKType::Pointer ifdk = FDKType::New();
  ifdk->SetInput( 0, tomographySource->GetOutput() );
  ifdk->SetInput( 1, slp->GetOutput() );
  ifdk->SetGeometry( geometry );
  ifdk->SetNumberOfIterations(3);
#ifdef USE_CUDA
  ifdk->SetForwardProjectionFilter(2);
  ifdk->SetBackProjectionFilter(2);
#else
  ifdk->SetForwardProjectionFilter(0);
  ifdk->SetBackProjectionFilter(0);
#endif
  TRY_AND_EXIT_ON_ITK_EXCEPTION( ifdk->Update() );

  // FOV
  typedef rtk::FieldOfViewImageFilter<OutputImageType, OutputImageType> FOVFilterType;
  FOVFilterType::Pointer fov=FOVFilterType::New();
  fov->SetInput(0, ifdk->GetOutput());
  fov->SetProjectionsStack(slp->GetOutput());
  fov->SetGeometry( geometry );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fov->Update() );

  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput(), 0.027, 27, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
