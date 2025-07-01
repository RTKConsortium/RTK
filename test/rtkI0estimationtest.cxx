#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkI0EstimationProjectionFilter.h"
#include <itkRandomImageSource.h>

/**
 * \file rtkI0estimationtest.cxx
 *
 * \brief Test rtk::I0EstimationProjectionFilter
 *
 * \author Sebastien Brousmiche
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using ImageType = itk::Image<unsigned short, Dimension>;

  using I0FilterType = rtk::I0EstimationProjectionFilter<ImageType, ImageType, 3>;
  auto i0est = I0FilterType::New();

  // Constant image sources
  auto                  size = itk::MakeSize(150, 150, 1);
  ImageType::RegionType region;
  region.SetSize(size);

  using RandomImageSourceType = itk::RandomImageSource<ImageType>;
  auto randomSource = RandomImageSourceType::New();
  randomSource->SetSize(size);

  i0est->SetExpectedI0(23);
  i0est->SetLambda(0.9);

  unsigned short minv = 0;
  unsigned short maxv = 0;
  for (unsigned int i = 0; i < 10; ++i)
  {
    minv = 3200 + 0 * i;
    maxv = 3800 + 400 * i;

    // std::cout << "Average = " << .5*(minv + maxv) << " - "<<minv <<" to "<<maxv<< std::endl;

    randomSource->SetMin(minv);
    randomSource->SetMax(maxv);
    randomSource->Update();

    i0est->SetInput(randomSource->GetOutput());
    i0est->Update();
  }

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
