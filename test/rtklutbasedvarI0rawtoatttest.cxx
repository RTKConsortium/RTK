#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkLUTbasedVariableI0RawToAttenuationImageFilter.h"
#include <itkImageRegionIterator.h>

/**
 * \file rtklutbasedvarI0rawtoatttest.cxx
 *
 * \brief Test rtk::LUTbasedVariableI0RawToAttenuationImageFilter
 *
 * \author Sebastien Brousmiche
 */

using ShortImageType = itk::Image<unsigned short, 2>;

void
fillImageWithRawData(ShortImageType::Pointer image, unsigned short I0)
{
  itk::ImageRegionIterator<ShortImageType> it(image, image->GetLargestPossibleRegion());
  it.GoToBegin();
  unsigned short i = 0;
  while (!it.IsAtEnd())
  {
    unsigned short I = (i % I0) + 1;
    it.Set(I);
    ++it;
    ++i;
  }
}

int
main(int, char **)
{
  auto convert = rtk::LUTbasedVariableI0RawToAttenuationImageFilter<ShortImageType, itk::Image<float, 2>>::New();

  // Constant image sources
  ShortImageType::RegionType region;
  region.SetSize(itk::MakeSize(10, 10));

  auto rawImage = ShortImageType::New();
  rawImage->SetRegions(region);
  rawImage->SetSpacing(itk::MakeVector(1.0, 1.0));
  rawImage->Allocate();

  convert->SetInput(rawImage);

  for (unsigned short i = 0; i < 10; ++i)
  {
    unsigned short I0 = 10 * i + 1;

    fillImageWithRawData(rawImage, I0);

    convert->SetI0(I0 / 2);
    convert->Update();
  }

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
