
#ifdef RAMP_FILTER_TEST_WITHOUT_FFTW
#  include "rtkConfiguration.h"
#  include <itkImageToImageFilter.h>
#  if defined(ITK_USE_FFTWF)
#    undef ITK_USE_FFTWF
#  endif
#  if defined(ITK_USE_FFTWD)
#    undef ITK_USE_FFTWD
#  endif
#  if defined(USE_FFTWF)
#    undef USE_FFTWF
#  endif
#  if defined(USE_FFTWD)
#    undef USE_FFTWD
#  endif
#endif

#include "rtkTest.h"
#include "rtkTestConfiguration.h"
#include "rtkMacro.h"

#include <math.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageFileWriter.h>

#include "rtkScatterGlareCorrectionImageFilter.h"

#include "rtkTestConfiguration.h"

/**
 * \file rtkscatterglarefiltertest.cxx
 *
 * \brief Functional test for the scatter glare correction filter
 *
 * \author Sebastien Brousmiche
 */

const unsigned Nprojections = 2;

const unsigned int Dimension = 3;
typedef itk::Image< float, Dimension >     ImageType;

ImageType::Pointer createInputImage(const std::vector<float> & coef)
{
  ImageType::SizeType size;
  size[0] = 650;
  size[1] = 700;
  size[2] = 1;

  ImageType::SpacingType spacing;
  spacing[0] = 0.296;
  spacing[1] = 0.296;
  spacing[2] = 1;

  ImageType::IndexType start;
  start[0] = 0;
  start[1] = 0;
  start[2] = 0;

  ImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(start);

  ImageType::Pointer inputI = ImageType::New();
  inputI->SetRegions(region);
  inputI->SetSpacing(spacing);
  inputI->Allocate();
  inputI->FillBuffer(1.0f);

  float a3 = coef[0];
  float b3 = coef[1];
  float b3sq = b3*b3;
  float dx = spacing[0];
  float dy = spacing[1];

  itk::ImageRegionIteratorWithIndex<ImageType> itK(inputI, inputI->GetLargestPossibleRegion());
  itK.GoToBegin();
  ImageType::IndexType idx;
  float sum = 0.0f;
  while (!itK.IsAtEnd()) {
    idx = itK.GetIndex();
    float xx = (float)idx[0] - (float)size[0] / 2.0f;
    float yy = (float)idx[1] - (float)size[1] / 2.0f;
    float rr2 = (xx*xx + yy*yy);
    float g = (a3*dx*dy / (2.0f * vnl_math::pi * b3sq)) * 1.0f / std::pow((1.0f + rr2 / b3sq), 1.5f);
    itK.Set(g);
    sum += g;
    ++itK;
  }

//  std::cout << "Input :" << sum << std::endl;
  
  return inputI;
}

int main(int , char** )
{  
  typedef rtk::ScatterGlareCorrectionImageFilter<ImageType, ImageType, float>   ScatterCorrectionType;
  ScatterCorrectionType::Pointer SFilter = ScatterCorrectionType::New();

  std::vector<float> coef;
  coef.push_back(0.0787f);
  coef.push_back(106.244f);


  SFilter->SetTruncationCorrection(0.5);
  SFilter->SetCoefficients(coef);

  ImageType::Pointer testImage = createInputImage(coef);
  
  for (unsigned i = 0; i < Nprojections; ++i) {
    SFilter->SetInput(testImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION( SFilter->Update() )

    ImageType::Pointer outputI = SFilter->GetOutput();
    itk::ImageRegionConstIterator<ImageType> itO(outputI, outputI->GetLargestPossibleRegion());
    itO.GoToBegin();
    float sum = 0.0f;
    while (!itO.IsAtEnd()) {
      sum += itO.Get();
      ++itO;
    }
 //   std::cout << "Output :" << sum << std::endl;
  }
 
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
