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

#ifndef __rtkTest_h
#define __rtkTest_h

#include <itkImageRegionConstIterator.h>
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

typedef rtk::ThreeDCircularProjectionGeometry GeometryType;


template<class TImage>
#if FAST_TESTS_NO_CHECKS
void CheckImageQuality(typename TImage::Pointer itkNotUsed(recon),
                       typename TImage::Pointer itkNotUsed(ref),
                       double itkNotUsed(ErrorPerPixelTolerance),
                       double itkNotUsed(PSNRTolerance)
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
  const double e           = 1e-10;
  const unsigned int nproj = g1->GetGantryAngles().size();
  if(g2->GetGantryAngles().size() != nproj)
    {
    std::cerr << "Unequal number of projections in the two geometries"
              << std::endl;
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
                      g2->GetProjectionOffsetsY()[i]) )
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

  typename TImage2::PixelType scalarProductGrads, scalarProductIms;
  scalarProductGrads = 0;
  scalarProductIms = 0;

  while( !itIm1A.IsAtEnd() )
    {
    scalarProductGrads += itIm1A.Get() * itIm1B.Get();
    ++itIm1A;
    ++itIm1B;
    }

  while( !itIm2A.IsAtEnd() )
    {
    scalarProductIms += itIm2A.Get() * itIm2B.Get();
    ++itIm2A;
    ++itIm2B;
    }

  // QI
  double ratio = scalarProductGrads / scalarProductIms;
  std::cout << "ratio = " << ratio << std::endl;

  // Checking results
  if (!(vcl_abs(ratio-1)<0.0001))
  {
    std::cerr << "Test Failed, ratio not valid! "
              << ratio << " instead of 1 +/- 0.0001" << std::endl;
    exit( EXIT_FAILURE);
  }
}
#endif

#endif //__rtkTest_h
