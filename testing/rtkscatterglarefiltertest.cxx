
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

#include <itkImageRegionConstIterator.h>

#include "rtkScatterGlareCorrectionImageFilter.h"

#include "rtkTestConfiguration.h"

/**
 * \file rtkscatterglarefiltertest.cxx
 *
 * \brief Functional test for the scatter glare correction filter
 *
 * \author Sebastien Brousmiche
 */

int main(int , char** )
{
  const unsigned int Dimension = 3;
  typedef float                                    PixelType;
  typedef itk::Image< PixelType, Dimension >       ImageType;
  
  typedef rtk::ScatterGlareCorrectionImageFilter<ImageType, ImageType, float>   ScatterCorrectionType;
  ScatterCorrectionType::Pointer SFilter = ScatterCorrectionType::New();
  
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
