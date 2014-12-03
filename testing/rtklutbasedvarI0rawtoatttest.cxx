#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkLUTbasedVariableI0RawToAttImageFilter.h"
#include <itkTimeProbe.h>
#include <itkImageRegionIterator.h>

/**
 * \file rtklutbasedvarI0rawtoatttest.cxx
 *
 * \brief Test rtk::LUTbasedVariableI0RawToAttImageFilter
 *
 * \author Sebastien Brousmiche
 */

typedef itk::Image<unsigned short, 2> ShortImageType;
typedef itk::Image<float, 2> FloatImageType;

void fillImageWithRawData(ShortImageType::Pointer image, unsigned short imageId)
{
  itk::ImageRegionIterator<ShortImageType> it(image, image->GetLargestPossibleRegion());
  it.GoToBegin();
  while (!it.IsAtEnd()){
    it.Set(imageId);
  }
}

int main(int, char** )
{
  itk::TimeProbe clock;

  typedef rtk::LUTbasedVariableI0RawToAttImageFilter ConvertFilterType;
  ConvertFilterType::Pointer convert = ConvertFilterType::New();
  
  // Constant image sources
  ShortImageType::SizeType size;
  size[0] = 2;
  size[1] = 2;
  size[2] = 1;
  ShortImageType::IndexType start;
  start.Fill(0);
  ShortImageType::RegionType region;
  region.SetIndex(start);
  region.SetSize(size);
  ShortImageType::SpacingType spacings;
  spacings[0] = 1.0;
  spacings[1] = 1.0;
  spacings[2] = 1.0;

  ShortImageType::Pointer rawImage = ShortImageType::New();
  rawImage->SetRegions(region);
  rawImage->SetSpacing(spacings);
  rawImage->Allocate();

  convert->SetInput(rawImage);

  for (unsigned short i = 0; i < 10; ++i)  
  {
    fillImageWithRawData(rawImage, i);

    convert->SetI0(i);

    clock.Start();

    convert->Update();

    clock.Stop();
  }

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
