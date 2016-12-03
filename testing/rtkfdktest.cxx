#include <itkImageRegionConstIterator.h>
#include <itkStreamingImageFilter.h>
#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  #include <itkImageRegionSplitterDirection.h>
#endif

#include "rtkTest.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkFieldOfViewImageFilter.h"

#ifdef USE_CUDA
#  include "rtkCudaFDKConeBeamReconstructionFilter.h"
#else
#  include "rtkFDKConeBeamReconstructionFilter.h"
#endif

/**
 * \file rtkfdktest.cxx
 *
 * \brief Functional test for classes performing FDK reconstructions
 *
 * This test generates the projections of a simulated Shepp-Logan phantom.
 * A CT image is reconstructed from each set of generated projection images
 * using the FDK algorithm and the reconstructed CT image is compared to the
 * expected results which is analytically computed.
 *
 * \author Simon Rit and Marc Vila
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

  std::cout << "\n\n****** Case 1: No streaming ******" << std::endl;

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
  typedef rtk::CudaFDKConeBeamReconstructionFilter                FDKType;
#else
  typedef rtk::FDKConeBeamReconstructionFilter< OutputImageType > FDKType;
#endif
  FDKType::Pointer feldkamp = FDKType::New();
  feldkamp->SetInput( 0, tomographySource->GetOutput() );
  feldkamp->SetInput( 1, slp->GetOutput() );
  feldkamp->SetGeometry( geometry );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );


  // FOV
  typedef rtk::FieldOfViewImageFilter<OutputImageType, OutputImageType> FOVFilterType;
  FOVFilterType::Pointer fov=FOVFilterType::New();
  fov->SetInput(0, feldkamp->GetOutput());
  fov->SetProjectionsStack(slp->GetOutput());
  fov->SetGeometry( geometry );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fov->Update() );

  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput(), 0.03, 26, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: perpendicular direction ******" << std::endl;

  ConstantImageSourceType::OutputImageType::DirectionType direction;
  direction[0][0] = 0;
  direction[0][1] = 1;
  direction[0][2] = 0;
  direction[1][0] = -1;
  direction[1][1] = 0;
  direction[1][2] = 0;
  direction[2][0] = 0;
  direction[2][1] = 0;
  direction[2][2] = 1;
  tomographySource->SetDirection(direction);
  origin[0] = -127.;
  origin[1] =  127.;
  origin[2] = -127.;
  tomographySource->SetOrigin( origin );
  fov->GetOutput()->ResetPipeline();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fov->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() )

  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput(), 0.03, 26, 2.0);
  std::cout << "Test PASSED! " << std::endl;

   std::cout << "\n\n****** Case 3: 45 degree tilt direction ******" << std::endl;

  direction[0][0] = 0.70710678118;
  direction[0][1] = -0.70710678118;
  direction[0][2] = 0.70710678118;
  direction[1][0] = 0.70710678118;
  direction[1][1] = 0;
  direction[1][2] = 0;
  direction[2][0] = 0;
  direction[2][1] = 0;
  direction[2][2] = 1;
  tomographySource->SetDirection(direction);
  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;
  tomographySource->SetOrigin( origin );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() )
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fov->Update() );

  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput(), 0.03, 26, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  std::cout << "\n\n****** Case 4: streaming ******" << std::endl;

  // Make sure that the data will be recomputed by releasing them
  fov->GetOutput()->ReleaseData();

  typedef itk::StreamingImageFilter<OutputImageType, OutputImageType> StreamingType;
  StreamingType::Pointer streamer = StreamingType::New();
  streamer->SetInput(0, fov->GetOutput());
  streamer->SetNumberOfStreamDivisions(8);
#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  itk::ImageRegionSplitterDirection::Pointer splitter = itk::ImageRegionSplitterDirection::New();
  splitter->SetDirection(2); // Prevent splitting along z axis. As a result, splitting will be performed along y axis
  streamer->SetRegionSplitter(splitter);
#endif
  TRY_AND_EXIT_ON_ITK_EXCEPTION( streamer->Update() );

  CheckImageQuality<OutputImageType>(streamer->GetOutput(), dsl->GetOutput(), 0.03, 26, 2.0);
  std::cout << "Test PASSED! " << std::endl;

  std::cout << "\n\n****** Case 5: small ROI ******" << std::endl;
  origin[0] = -5.;
  origin[1] = -13.;
  origin[2] = -20.;
  size[0] = 64;
  size[1] = 64;
  size[2] = 64;
  tomographySource->SetOrigin( origin );
  tomographySource->SetSize( size );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( fov->UpdateLargestPossibleRegion() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->UpdateLargestPossibleRegion() )
  CheckImageQuality<OutputImageType>(fov->GetOutput(), dsl->GetOutput(), 0.03, 26, 2.0);
  std::cout << "Test PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
