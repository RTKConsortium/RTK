#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkI0EstimationProjectionFilter.h"
#include <itkRandomImageSource.h>
#include <itkTimeProbe.h>

/**
 * \file rtkI0estimationtest.cxx
 *
 * \brief Test rtk::I0EstimationProjectionFilter
 *
 * \author Sebastien Brousmiche
 */

int main(int, char** )
{
  itk::TimeProbe clock;

  const unsigned int Dimension = 3;
  typedef itk::Image<unsigned short, Dimension> ImageType;

  typedef rtk::I0EstimationProjectionFilter<ImageType, ImageType, 3> I0FilterType;
  I0FilterType::Pointer i0est = I0FilterType::New();

  // Constant image sources
  ImageType::SizeType size;
  size[0] = 150;
  size[1] = 150;
  size[2] = 1;
  ImageType::IndexType start;
  start.Fill(0);
  ImageType::RegionType region;
  region.SetIndex(start);
  region.SetSize(size);

  typedef itk::RandomImageSource< ImageType > RandomImageSourceType;
  RandomImageSourceType::Pointer randomSource = RandomImageSourceType::New();
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

    clock.Start();

    i0est->Update();

    clock.Stop();
  }

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
