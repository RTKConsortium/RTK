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

#include "rtkFDKConeBeamReconstructionFilter.h"
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include "rtkFieldOfViewImageFilter.h"
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
        //std::cout << "Not equal" << std::endl;
        TestError += vcl_abs(RefVal - TestVal);
        //std::cout << "Error ="<< TestError << std::endl;
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
  const unsigned int NumberOfProjectionImages = 360;

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  // FOV filter Input Volume, it is used as the input to create the fov mask.
  const ConstantImageSourceType::Pointer fovInput = ConstantImageSourceType::New();
  origin[0] = -127;
  origin[1] = -127;
  origin[2] = -127;
  size[0] = 256;
  size[1] = 256;
  size[2] = 256;
  spacing[0] = 1.;
  spacing[1] = 1.;
  spacing[2] = 1.;
  fovInput->SetOrigin( origin );
  fovInput->SetSpacing( spacing );
  fovInput->SetSize( size );
  fovInput->SetConstant( 1. );

  // FDK Input Projections, it is used as the input to create the fov mask.
  const ConstantImageSourceType::Pointer fdkInput = ConstantImageSourceType::New();
  origin[0] = -127;
  origin[1] = -127;
  origin[2] = -127;
  size[0] = 256;
  size[1] = 256;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 1.;
  spacing[1] = 1.;
  spacing[2] = 1.;
  fdkInput->SetOrigin( origin );
  fdkInput->SetSpacing( spacing );
  fdkInput->SetSize( size );
  fdkInput->SetConstant( 1. );

  // Stack of empty projections
  const ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -127;
  origin[1] = -127;
  origin[2] = -127;
  size[0] = 256;
  size[1] = 256;
  size[2] = 256;
  spacing[0] = 1.;
  spacing[1] = 1.;
  spacing[2] = 1.;
  projectionsSource->SetOrigin( origin );
  projectionsSource->SetSpacing( spacing );
  projectionsSource->SetSize( size );
  projectionsSource->SetConstant( 0. );

  //FOV filter
  typedef rtk::FieldOfViewImageFilter<OutputImageType, OutputImageType> FOVFilterType;
  FOVFilterType::Pointer fov=FOVFilterType::New();

  // FDK reconstruction filtering
  typedef rtk::FDKConeBeamReconstructionFilter< OutputImageType > FDKCPUType;
  FDKCPUType::Pointer feldkamp = FDKCPUType::New();

  // Writer
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();

  // Geometry
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  // Creating geometry
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(1000, 1500., noProj*360./NumberOfProjectionImages);

  // Calculating ray box intersection of the resized Input
  fov->SetInput(0, fovInput->GetOutput());
  fov->SetProjectionsStack(projectionsSource->GetOutput());
  fov->SetGeometry( geometry );
  fov->Update();

  // Writting
  writer->SetFileName( "fov.mhd" );
  writer->SetInput( fov->GetOutput() );
  writer->Update();

  // Backproject stack of projections
  feldkamp->SetInput( 0, projectionsSource->GetOutput() );
  feldkamp->SetInput( 1, fdkInput->GetOutput() );
  feldkamp->SetGeometry( geometry );
  feldkamp->GetRampFilter()->SetHannCutFrequency(0.);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );

//  typedef itk::ImageRegionIterator<OutputImageType> OutputIterator;
//  typedef typename OutputImageType::RegionType OutputImageRegionType;
//  const OutputImageRegionType outputRegion = feldkamp->GetOutput()->GetLargestPossibleRegion();
//  OutputIterator itOut(feldkamp->GetOutput(), outputRegion );
//  itOut.GoToBegin();

//  const OutputImageType::SizeValueType loopSize = outputRegion.GetSize(0)*outputRegion.GetSize(1)*outputRegion.GetSize(2);
//  bool inside = false;
//  OutputPixelType previous = 0.;

//  for(unsigned int k=0; k<loopSize; k++)
//    {
//      if(inside)
//        {
//          if(itOut.Get()-previous > 0.01)
//            inside = false;
//          previous = itOut.Get();
//          itOut.Set(1.);
//        }
//      else
//        {
//          if(itOut.Get()-previous < -0.01)
//            inside = true;
//          previous = itOut.Get();
//          itOut.Set(0.);
//        }
//      ++itOut;
//    }
  // Writting
  writer->SetFileName( "bkp_fov.mhd" );
  writer->SetInput( feldkamp->GetOutput() );
  writer->Update();

  CheckImageQuality<OutputImageType>(fov->GetOutput(), feldkamp->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  exit(EXIT_SUCCESS);
}
