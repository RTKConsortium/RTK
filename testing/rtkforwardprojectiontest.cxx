/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include "rtkJosephForwardProjectionImageFilter.h"
#include <itkExtractImageFilter.h>
#include <itkMultiplyByConstantImageFilter.h>

template<class TImage>
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

    if( TestVal != RefVal )
      {
        std::cout << "Not equal" << std::endl;
        TestError += vcl_abs(RefVal - TestVal);
        std::cout << "Error ="<< TestError << std::endl;
        EnerError += vcl_pow(ErrorType(RefVal - TestVal), 2.);
      }
    ++itTest;
    ++itRef;
    }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError/recon->GetBufferedRegion().GetNumberOfPixels();
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
  if (ErrorPerPixel > 0.005)
  {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 0.005" << std::endl;
    exit( EXIT_FAILURE);
  }
  if (PSNR < 25.)
  {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 25" << std::endl;
    exit( EXIT_FAILURE);
  }
}

int main(int argc, char* argv[])
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  const unsigned int NumberOfProjectionImages = 1;

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer source[Dimension];

  source[0] = ConstantImageSourceType::New();
  origin[0] = -127;
  origin[1] = -127;
  origin[2] = -127;
  size[0] = 256;
  size[1] = 256;
  size[2] = 176;
  spacing[0] = 1.;
  spacing[1] = 1.;
  spacing[2] = 1.;
  source[0]->SetOrigin( origin );
  source[0]->SetSpacing( spacing );
  source[0]->SetSize( size );
  source[0]->SetConstant( 1. );

  //ConstantImageSourceType::Pointer fullSource = ConstantImageSourceType::New();
  source[1] = ConstantImageSourceType::New();
  origin[0] = -127;
  origin[1] = -127;
  origin[2] = -127;
  size[0] = 256;
  size[1] = 256;
  size[2] = 256;
  spacing[0] = 1.;
  spacing[1] = 1.;
  spacing[2] = 1.;
  source[1]->SetOrigin( origin );
  source[1]->SetSpacing( spacing );
  source[1]->SetSize( size );
  source[1]->SetConstant( 1. );

  //ConstantImageSourceType::Pointer fullSource = ConstantImageSourceType::New();
  source[2] = ConstantImageSourceType::New();
  origin[0] = -127;
  origin[1] = -127;
  origin[2] = -127;
  size[0] = 256;
  size[1] = 256;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 1.;
  spacing[1] = 1.;
  spacing[2] = 1.;
  source[2]->SetOrigin( origin );
  source[2]->SetSpacing( spacing );
  source[2]->SetSize( size );
  source[2]->SetConstant( 0. );

  // Create projection image filter
  typedef rtk::RayBoxIntersectionImageFilter<OutputImageType, OutputImageType> RBIType;
  typedef rtk::JosephForwardProjectionImageFilter< OutputImageType, OutputImageType >       JosephForwardProjectionType;
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  RBIType::Pointer rbi = RBIType::New();
  JosephForwardProjectionType::Pointer jfp = JosephForwardProjectionType::New();
  WriterType::Pointer writer = WriterType::New();
  GeometryType::Pointer geometry = GeometryType::New();
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(48.5, 1000., noProj);
  source[0]->UpdateOutputInformation();
  rbi->SetInput( source[2]->GetOutput() );
  rbi->SetBoxFromImage( source[0]->GetOutput() );
  rbi->SetGeometry( geometry );
  rbi->Update();
  // Write
  writer->SetFileName( "rbi.mhd" );
  writer->SetInput( rbi->GetOutput() );
  writer->Update();
  jfp->SetInput(source[2]->GetOutput());
  jfp->SetInput( 1, source[1]->GetOutput() );
  jfp->SetGeometry( geometry );
  jfp->SetNumberOfThreads(1);
  jfp->Update();
  writer->SetFileName( "Joseph.mhd" );
  writer->SetInput( jfp->GetOutput() );
  writer->Update();
  CheckImageQuality<OutputImageType>(rbi->GetOutput(), jfp->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;
  exit(EXIT_SUCCESS);
}
