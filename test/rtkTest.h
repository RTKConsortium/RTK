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

#ifndef rtkTest_h
#define rtkTest_h

#include <itkImageRegionConstIterator.h>
#include <itkImageFileWriter.h>
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkTestConfiguration.h"

typedef rtk::ThreeDCircularProjectionGeometry GeometryType;


template<class TImage>
#if FAST_TESTS_NO_CHECKS
void CheckImageQuality(typename TImage::Pointer itkNotUsed(recon),
                       typename TImage::Pointer itkNotUsed(ref),
                       double itkNotUsed(ErrorPerPixelTolerance),
                       double itkNotUsed(PSNRTolerance),
                       double itkNotUsed(RefValueForPSNR))
{
}
#else
void CheckImageQuality(typename TImage::Pointer recon,
                       typename TImage::Pointer ref,
                       double ErrorPerPixelTolerance,
                       double PSNRTolerance,
                       double RefValueForPSNR)
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
  ErrorType PSNR = 20*log10(RefValueForPSNR) - 10*log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (RefValueForPSNR-ErrorPerPixel)/RefValueForPSNR;
  std::cout << "QI = " << QI << std::endl;

//   // It is often necessary to write the images and look at them
//   // to understand why a given test fails. This portion of code
//   // does that. It should be left here but commented out, since
//   // it is only useful in specific debugging tasks
//   typedef itk::ImageFileWriter<TImage> FileWriterType;
//   typename FileWriterType::Pointer writer = FileWriterType::New();
//   writer->SetInput(recon);
//   writer->SetFileName("Reconstruction.mhd");
//   writer->Update();
//   writer->SetInput(ref);
//   writer->SetFileName("Reference.mhd");
//   writer->Update();
//   // End of results writing

  // Checking results. As a comparison with NaN always returns false,
  // this design allows to detect NaN results and cause test failure
  if (!(ErrorPerPixel < ErrorPerPixelTolerance))
    {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of "<< ErrorPerPixelTolerance << std::endl;
    exit( EXIT_FAILURE);
    }
  if (!(PSNR > PSNRTolerance))
    {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of " << PSNRTolerance << std::endl;
    exit( EXIT_FAILURE);
    }
}
#endif //FAST_TESTS_NO_CHECKS

template<class TImage>
#if FAST_TESTS_NO_CHECKS
void CheckVectorImageQuality(typename TImage::Pointer itkNotUsed(recon),
                             typename TImage::Pointer itkNotUsed(ref),
                             double itkNotUsed(ErrorPerPixelTolerance),
                             double itkNotUsed(PSNRTolerance),
                             double itkNotUsed(RefValueForPSNR))
{
}
#else
void CheckVectorImageQuality(typename TImage::Pointer recon,
                             typename TImage::Pointer ref,
                             double ErrorPerPixelTolerance,
                             double PSNRTolerance,
                             double RefValueForPSNR)
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
    TestError += (RefVal - TestVal).GetNorm();
    EnerError += (RefVal - TestVal).GetSquaredNorm();
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
  ErrorType PSNR = 20*log10(RefValueForPSNR) - 10*log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (RefValueForPSNR-ErrorPerPixel)/RefValueForPSNR;
  std::cout << "QI = " << QI << std::endl;

//   // It is often necessary to write the images and look at them
//   // to understand why a given test fails. This portion of code
//   // does that. It should be left here but commented out, since
//   // it is only useful in specific debugging tasks
//   typedef itk::ImageFileWriter<TImage> FileWriterType;
//   typename FileWriterType::Pointer writer = FileWriterType::New();
//   writer->SetInput(recon);
//   writer->SetFileName("Reconstruction.mhd");
//   writer->Update();
//   writer->SetInput(ref);
//   writer->SetFileName("Reference.mhd");
//   writer->Update();
//   // End of results writing

  // Checking results. As a comparison with NaN always returns false,
  // this design allows to detect NaN results and cause test failure
  if (!(ErrorPerPixel < ErrorPerPixelTolerance))
    {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of "<< ErrorPerPixelTolerance << std::endl;
    exit( EXIT_FAILURE);
    }
  if (!(PSNR > PSNRTolerance))
    {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of " << PSNRTolerance << std::endl;
    exit( EXIT_FAILURE);
    }
}
#endif //FAST_TESTS_NO_CHECKS


template<class TImage>
#if FAST_TESTS_NO_CHECKS
void CheckVariableLengthVectorImageQuality(typename TImage::Pointer itkNotUsed(recon),
                                           typename TImage::Pointer itkNotUsed(ref),
                                           double itkNotUsed(ErrorPerPixelTolerance),
                                           double itkNotUsed(PSNRTolerance),
                                           double itkNotUsed(RefValueForPSNR))
{
}
#else
void CheckVariableLengthVectorImageQuality(typename TImage::Pointer recon,
                                           typename TImage::Pointer ref,
                                           double ErrorPerPixelTolerance,
                                           double PSNRTolerance,
                                           double RefValueForPSNR)
{
  if (!(recon->GetVectorLength() == ref->GetVectorLength()))
    {
    std::cerr << "Test Failed, image's vector length is "
              << recon->GetVectorLength() << " instead of "<< ref->GetVectorLength() << std::endl;
    exit( EXIT_FAILURE);
    }

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
    typename TImage::PixelType TestVec = itTest.Get();
    typename TImage::PixelType RefVec = itRef.Get();
    double accumulatedError = 0;
    for (unsigned int channel=0; channel < ref->GetVectorLength(); channel++)
      {
      accumulatedError += (RefVec[channel] - TestVec[channel]) * (RefVec[channel] - TestVec[channel]);
      }
    TestError += sqrt(accumulatedError);
    EnerError += accumulatedError;
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
  ErrorType PSNR = 20*log10(RefValueForPSNR) - 10*log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (RefValueForPSNR-ErrorPerPixel)/RefValueForPSNR;
  std::cout << "QI = " << QI << std::endl;

//   // It is often necessary to write the images and look at them
//   // to understand why a given test fails. This portion of code
//   // does that. It should be left here but commented out, since
//   // it is only useful in specific debugging tasks
//   typedef itk::ImageFileWriter<TImage> FileWriterType;
//   typename FileWriterType::Pointer writer = FileWriterType::New();
//   writer->SetInput(recon);
//   writer->SetFileName("Reconstruction.mhd");
//   writer->Update();
//   writer->SetInput(ref);
//   writer->SetFileName("Reference.mhd");
//   writer->Update();
//   // End of results writing

  // Checking results. As a comparison with NaN always returns false,
  // this design allows to detect NaN results and cause test failure
  if (!(ErrorPerPixel < ErrorPerPixelTolerance))
    {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of "<< ErrorPerPixelTolerance << std::endl;
    exit( EXIT_FAILURE);
    }
  if (!(PSNR > PSNRTolerance))
    {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of " << PSNRTolerance << std::endl;
    exit( EXIT_FAILURE);
    }
}
#endif //FAST_TESTS_NO_CHECKS

void CheckGeometries(GeometryType *g1, GeometryType *g2)
{
//  // It is often necessary to write the geometries and look at them
//  // to understand why a given test fails. This portion of code
//  // does that. It should be left here but commented out, since
//  // it is only useful in specific debugging tasks
//  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter =
//    rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
//  xmlWriter->SetFilename("g1.xml");
//  xmlWriter->SetObject(g1);
//  xmlWriter->WriteFile();
//  xmlWriter->SetFilename("g2.xml");
//  xmlWriter->SetObject(g2);
//  xmlWriter->WriteFile();
//  // End of results writing

  const double e           = 1e-10;
  const unsigned int nproj = g1->GetGantryAngles().size();
  if(g2->GetGantryAngles().size() != nproj)
    {
    std::cerr << "Unequal number of projections in the two geometries"
              << std::endl;
    exit(EXIT_FAILURE);
    }
  if(e < std::fabs(g1->GetRadiusCylindricalDetector() -
                   g2->GetRadiusCylindricalDetector()) )
    {
    std::cerr << "Geometries don't have the same cylindrical detector radius" << std::endl;
    exit(EXIT_FAILURE);
    }

  for(unsigned int i=0; i<nproj; i++)
    {
    if( e < rtk::ThreeDCircularProjectionGeometry::ConvertAngleBetween0And2PIRadians(
              std::fabs(g1->GetGantryAngles()[i] -
                        g2->GetGantryAngles()[i])) ||
        e < rtk::ThreeDCircularProjectionGeometry::ConvertAngleBetween0And2PIRadians(
              std::fabs(g1->GetOutOfPlaneAngles()[i] -
                        g2->GetOutOfPlaneAngles()[i])) ||
        e < rtk::ThreeDCircularProjectionGeometry::ConvertAngleBetween0And2PIRadians(
              std::fabs(g1->GetInPlaneAngles()[i] -
                        g2->GetInPlaneAngles()[i])) ||
        e < std::fabs(g1->GetSourceToIsocenterDistances()[i] -
                      g2->GetSourceToIsocenterDistances()[i]) ||
        e < std::fabs(g1->GetSourceOffsetsX()[i] -
                      g2->GetSourceOffsetsX()[i]) ||
        e < std::fabs(g1->GetSourceOffsetsY()[i] -
                      g2->GetSourceOffsetsY()[i]) ||
        e < std::fabs(g1->GetSourceToDetectorDistances()[i] -
                      g2->GetSourceToDetectorDistances()[i]) ||
        e < std::fabs(g1->GetProjectionOffsetsX()[i] -
                      g2->GetProjectionOffsetsX()[i]) ||
        e < std::fabs(g1->GetProjectionOffsetsY()[i] -
                      g2->GetProjectionOffsetsY()[i]) ||
        e < std::fabs(g1->GetCollimationUInf()[i] -
                      g2->GetCollimationUInf()[i]) ||
        e < std::fabs(g1->GetCollimationVInf()[i] -
                      g2->GetCollimationVInf()[i]) ||
        e < std::fabs(g1->GetCollimationUSup()[i] -
                      g2->GetCollimationUSup()[i]) ||
        e < std::fabs(g1->GetCollimationVSup()[i] -
                      g2->GetCollimationVSup()[i]) )
      {
      std::cerr << "Geometry of projection #" << i << " is unvalid."
                << std::endl;
      exit(EXIT_FAILURE);
      }

    }
}


template<class TImage1, class TImage2>
#if FAST_TESTS_NO_CHECKS
void CheckScalarProducts(typename TImage1::Pointer itkNotUsed(im1A),
                         typename TImage1::Pointer itkNotUsed(im1B),
                         typename TImage2::Pointer itkNotUsed(im2A),
                         typename TImage2::Pointer itkNotUsed(im2B))
{
}
#else
void CheckScalarProducts(typename TImage1::Pointer im1A,
                         typename TImage1::Pointer im1B,
                         typename TImage2::Pointer im2A,
                         typename TImage2::Pointer im2B)
{
  typedef itk::ImageRegionConstIterator<TImage1> Image1IteratorType;
  typedef itk::ImageRegionConstIterator<TImage2> Image2IteratorType;
  Image1IteratorType itIm1A( im1A, im1A->GetLargestPossibleRegion() );
  Image1IteratorType itIm1B( im1B, im1B->GetLargestPossibleRegion() );
  Image2IteratorType itIm2A( im2A, im2A->GetLargestPossibleRegion() );
  Image2IteratorType itIm2B( im2B, im2B->GetLargestPossibleRegion() );

  typename TImage2::PixelType scalarProductT1, scalarProductT2;
  scalarProductT1 = 0;
  scalarProductT2 = 0;

  while( !itIm1A.IsAtEnd() )
    {
    scalarProductT1 += itIm1A.Get() * itIm1B.Get();
    ++itIm1A;
    ++itIm1B;
    }

  while( !itIm2A.IsAtEnd() )
    {
    scalarProductT2 += itIm2A.Get() * itIm2B.Get();
    ++itIm2A;
    ++itIm2B;
    }

  // QI
  double ratio = scalarProductT1 / scalarProductT2;
  std::cout << "1 - ratio = " << 1 - ratio << std::endl;

//  // It is often necessary to write the images and look at them
//  // to understand why a given test fails. This portion of code
//  // does that. It should be left here but commented out, since
//  // it is only useful in specific debugging tasks
//  typedef itk::ImageFileWriter<TImage1> FileWriterType1;
//  typename FileWriterType1::Pointer writer = FileWriterType1::New();
//  writer->SetInput(im1A);
//  writer->SetFileName("im1A.mhd");
//  writer->Update();
//  writer->SetInput(im1B);
//  writer->SetFileName("im1B.mhd");
//  writer->Update();

//  typedef itk::ImageFileWriter<TImage2> FileWriterType2;
//  typename FileWriterType2::Pointer writer2 = FileWriterType2::New();
//  writer2->SetInput(im2A);
//  writer2->SetFileName("im2A.mhd");
//  writer2->Update();
//  writer2->SetInput(im2B);
//  writer2->SetFileName("im2B.mhd");
//  writer2->Update();
//  // End of results writing


  // Checking results
  if (!(vcl_abs(ratio-1)<0.001))
  {
    std::cerr << "Test Failed, ratio not valid! "
              << ratio << " instead of 1 +/- 0.001" << std::endl;
    exit( EXIT_FAILURE);
  }
}
#endif

#endif //rtkTest_h
