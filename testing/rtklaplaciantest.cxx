#include "itkRandomImageSource.h"
#include "rtkLaplacianImageFilter.h"
#include "rtkMacro.h"
#include "rtkTest.h"
#include "rtkDrawSheppLoganFilter.h"
#include "itkImageFileReader.h"
#include "rtkConstantImageSource.h"

/**
 * \file rtklaplaciantest.cxx
 *
 * \brief Tests whether the computation of the laplacian actually works
 *
 * This test generates a random volume and computes its laplacian.
 *
 * \author Cyril Mory
 */

int main(int, char** )
{
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

#ifdef USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension >     OutputImageType;
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
  size[0] = 32;
  size[1] = 32;
  size[2] = 32;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
#endif
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  // Generate a shepp logan phantom
  typedef rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType> DSLType;
  DSLType::Pointer dsl=DSLType::New();
  dsl->SetInput( tomographySource->GetOutput() );
  dsl->SetPhantomScale(128.);
  dsl->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() );

  // Create and set the laplacian filter
  typedef rtk::LaplacianImageFilter<OutputImageType>                LaplacianFilterType;
  LaplacianFilterType::Pointer laplacian = LaplacianFilterType::New();
  laplacian->SetInput(dsl->GetOutput());

  // Compute the laplacian of the shepp logan
  TRY_AND_EXIT_ON_ITK_EXCEPTION( laplacian->Update() );

  // Read a reference image
  typedef itk::ImageFileReader<OutputImageType> ReaderType;
  ReaderType::Pointer readerRef = ReaderType::New();
  readerRef->SetFileName( std::string(RTK_DATA_ROOT) +
                          std::string("/Baseline/Laplacian/Laplacian.mha"));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // Compare the result with the reference
  CheckImageQuality<OutputImageType>(laplacian->GetOutput(), readerRef->GetOutput(), 0.00001, 15, 2.0);

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
