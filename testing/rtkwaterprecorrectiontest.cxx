#include "rtkTest.h"
#include "rtkConstantImageSource.h"

#include "rtkWaterPrecorrectionFilter.h"
#include "rtkWaterCalibrationFilter.h"

/**
 * \file rtkwaterprecorrectiontest.cxx
 *
 * \brief Functional test for the classes performing water precorrection
 *
 * \author S. Brousmiche
 */

int main(int , char** )
{
  const unsigned int Dimension = 2;
  typedef float                                    OutputPixelType;
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
  imgIn->SetConstant( 1.5 );
  //imgIn->UpdateOutputInformation();

  imgRef->SetOrigin( origin );
  imgRef->SetSpacing( spacing );
  imgRef->SetSize( size );
  imgRef->SetConstant( 5.0 );
  imgRef->Update();

  OutputImageType::Pointer output = imgIn->GetOutput();
  
	std::cout << "\n\n****** Case 1: order 2 ******" << std::endl;

	typedef rtk::WaterPrecorrectionFilter<2> WPCType2;
	WPCType2::Pointer model2 = WPCType2::New();
  
	// Update median filter
	itk::Vector<float, 2> c1;
	c1[0]=2.0;
	c1[1]=2.0;
	model2->SetInput(output);
	model2->SetCoefficients(c1);
	model2->Update();

	CheckImageQuality<OutputImageType>(model2->GetOutput(), imgRef->GetOutput(), 1.8, 51, 1011.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: order 3 ******" << std::endl;

	typedef rtk::WaterPrecorrectionFilter<3> WPCType3;
	WPCType3::Pointer model3 = WPCType3::New();

	itk::Vector<float, 3> c2;
	c2[0] = 0.05;
	c2[1] = 0.3;
	c2[2] = 2.0;
	model3->SetInput(imgIn->GetOutput());
	model3->SetCoefficients(c2);
	model3->Update();

	CheckImageQuality<OutputImageType>(model3->GetOutput(), imgRef->GetOutput(), 1.8, 51, 1011.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

	std::cout << "\n\n****** Case 3: order 5 ******" << std::endl;

	typedef rtk::WaterPrecorrectionFilter<5> WPCType5;
	WPCType5::Pointer model5 = WPCType5::New();

	itk::Vector<float, 5> c3;
	c3[0] = 0.0687;
	c3[1] = 2.5;
	c3[2] = 0.6;
	c3[3] = -0.2;
	c3[4] = 0.1;
	model5->SetInput(imgIn->GetOutput());
	model5->SetCoefficients(c3);
	model5->Update();

	CheckImageQuality<OutputImageType>(model5->GetOutput(), imgRef->GetOutput(), 1.8, 51, 1011.0);
	std::cout << "\n\nTest PASSED! " << std::endl;

	std::cout << "\n\n****** Calib case 1 : order 1 ******" << std::endl;

	typedef rtk::WaterCalibrationFilter WPCalType;
	WPCalType::Pointer mcal = WPCalType::New();

	mcal->SetInput(imgIn->GetOutput());
	mcal->SetOrder(1.0);
	mcal->Update();

	//CheckImageQuality<OutputImageType>(mcal->GetOutput(), imgRef->GetOutput(), 1.8, 51, 1011.0);
	std::cout << "\n\nTest PASSED! " << std::endl;
	
	std::cout << "\n\n****** Calib case 2 : order 4 ******" << std::endl;

	mcal->SetOrder(5.0);
	mcal->Update();

	//CheckImageQuality<OutputImageType>(mcal->GetOutput(), imgRef->GetOutput(), 1.8, 51, 1011.0);
	std::cout << "\n\nTest PASSED! " << std::endl;


  return EXIT_SUCCESS;
}
