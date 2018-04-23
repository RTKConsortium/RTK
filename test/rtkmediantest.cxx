#include "rtkTest.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkAdditiveGaussianNoiseImageFilter.h"

#include "rtkMedianImageFilter.h"

/**
 * \file rtkmediantest.cxx
 *
 * \brief Functional test for the classes performing median filtering
 *
 * This test perfoms a median filtering on a 2D image with the presence
 * of Gaussian noise and using a window of 3x3 and 3x2. Compares
 * the obtained result with a reference image previously calculated.
 *
 * \author Marc Vila
 */

int main(int , char** )
{
  const unsigned int Dimension = 2;
  typedef unsigned short                           OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  // Create constant image of value 2 and reference image.
  ConstantImageSourceType::Pointer imgIn  = ConstantImageSourceType::New();
  ConstantImageSourceType::Pointer imgRef = ConstantImageSourceType::New();

  origin[0] = -126;
  origin[1] = -126;
  size[0] = 16;
  size[1] = 16;
  spacing[0] = 16.;
  spacing[1] = 16.;

  imgIn->SetOrigin( origin );
  imgIn->SetSpacing( spacing );
  imgIn->SetSize( size );
  imgIn->SetConstant( 1000 );
  //imgIn->UpdateOutputInformation();

  imgRef->SetOrigin( origin );
  imgRef->SetSpacing( spacing );
  imgRef->SetSize( size );
  imgRef->SetConstant( 1000 );
  imgRef->Update();

  // Add noise to Input Image
  OutputImageType::Pointer output = imgIn->GetOutput();
  typedef rtk::AdditiveGaussianNoiseImageFilter< OutputImageType > NIFType;
  NIFType::Pointer noisy=NIFType::New();
  noisy->SetInput( imgIn->GetOutput() );
  noisy->SetMean( 0 );
  noisy->SetStandardDeviation( 5 );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( noisy->Update() );
  output = noisy->GetOutput();

  // Median filter
  typedef rtk::MedianImageFilter MEDType;
  MEDType::Pointer median = MEDType::New();

  std::cout << "\n\n****** Case 1: median 3x3 ******" << std::endl;

  // Update median filter
  itk::Vector<unsigned int,2> median_window;
  median_window[0]=3;
  median_window[1]=3;
  median->SetInput(output);
  median->SetMedianWindow(median_window);
  median->Update();

  CheckImageQuality<OutputImageType>(median->GetOutput(), imgRef->GetOutput(), 1.8, 51, 1011.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: median 3x2 ******" << std::endl;

  // Update median filter
  median_window[0]=3;
  median_window[1]=2;
  median->SetInput( imgIn->GetOutput() );
  median->SetMedianWindow(median_window);
  median->Update();

  CheckImageQuality<OutputImageType>(median->GetOutput(), imgRef->GetOutput(), 1.8, 51, 1011.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
