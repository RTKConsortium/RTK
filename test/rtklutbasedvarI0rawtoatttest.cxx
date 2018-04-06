#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkLUTbasedVariableI0RawToAttenuationImageFilter.h"
#include <itkTimeProbe.h>
#include <itkImageRegionIterator.h>

/**
 * \file rtklutbasedvarI0rawtoatttest.cxx
 *
 * \brief Test rtk::LUTbasedVariableI0RawToAttenuationImageFilter
 *
 * \author Sebastien Brousmiche
 */

typedef itk::Image<unsigned short, 2> ShortImageType;
typedef itk::Image<float, 2> FloatImageType;

void fillImageWithRawData(ShortImageType::Pointer image, unsigned short I0)
{
  itk::ImageRegionIterator< ShortImageType > it(image, image->GetLargestPossibleRegion());
  it.GoToBegin();
  unsigned short i = 0;
  while (!it.IsAtEnd()){
    unsigned short I = (i%I0)+1;
    it.Set(I);
    ++it;
    ++i;
  }
}

int main(int, char** )
{
  itk::TimeProbe clock;

  typedef rtk::LUTbasedVariableI0RawToAttenuationImageFilter<ShortImageType, FloatImageType> ConvertFilterType;
  ConvertFilterType::Pointer convert = ConvertFilterType::New();

  // Constant image sources
  ShortImageType::SizeType size;
  size[0] = 10;
  size[1] = 10;
  ShortImageType::IndexType start;
  start.Fill(0);
  ShortImageType::RegionType region;
  region.SetIndex(start);
  region.SetSize(size);
  ShortImageType::SpacingType spacings;
  spacings[0] = 1.0;
  spacings[1] = 1.0;

  ShortImageType::Pointer rawImage = ShortImageType::New();
  rawImage->SetRegions(region);
  rawImage->SetSpacing(spacings);
  rawImage->Allocate();

  convert->SetInput(rawImage);

  for (unsigned short i = 0; i < 10; ++i)
  {
    unsigned short I0 = 10 * i + 1;

    fillImageWithRawData(rawImage, I0);

    convert->SetI0(I0/2);

    clock.Start();

    convert->Update();

    clock.Stop();
  }

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
