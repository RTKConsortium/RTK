#include "rtkTest.h"
#include "rtkI0EstimationProjectionFilter.h"
#include "rtkMacro.h"
#include <itkImageRegionIterator.h>
#include <itkRandomImageSource.h>

#include <vector>

/**
 * \file rtkI0estimationtest.cxx
 *
 * \brief Test rtk::I0EstimationProjectionFilter
 *
 * \author Sebastien Brousmiche
 */

int
rtkI0estimationtest(int, char *[])
{
  constexpr unsigned int Dimension = 3;
  using ImageType = itk::Image<unsigned short, Dimension>;

  auto i0est = rtk::I0EstimationProjectionFilter<ImageType, ImageType, 3>::New();

  // Constant image sources
  auto                  size = itk::MakeSize(150, 150, 1);
  ImageType::RegionType region;
  region.SetSize(size);

  auto randomSource = itk::RandomImageSource<ImageType>::New();
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

  // Regression: the I0 estimate must not depend on the number of work units.
  // ITK may split the image in fewer regions than requested work units. The
  // estimate used to rely on the last of exactly m_Nthreads region merges,
  // so those bounds were skipped and a garbage I0 was silently produced on
  // machines requesting many work units.
  minv = 3200;
  maxv = 4200;
  randomSource->SetMin(minv);
  randomSource->SetMax(maxv);
  randomSource->Update();

  // Keep a pristine copy of the input: the filter is in-place capable and
  // may steal the buffer of the image it is fed.
  auto pristine = ImageType::New();
  pristine->CopyInformation(randomSource->GetOutput());
  pristine->SetBufferedRegion(randomSource->GetOutput()->GetBufferedRegion());
  pristine->Allocate();
  itk::ImageRegionConstIterator<ImageType> itSrc(randomSource->GetOutput(), pristine->GetBufferedRegion());
  itk::ImageRegionIterator<ImageType>      itDst(pristine, pristine->GetBufferedRegion());
  for (itSrc.GoToBegin(), itDst.GoToBegin(); !itSrc.IsAtEnd(); ++itSrc, ++itDst)
  {
    itDst.Set(itSrc.Get());
  }

  const std::vector<unsigned int> workUnitCounts = { 1,   3,   7,   13,  23,  37,   53,   64,   89,
                                                     127, 150, 151, 200, 500, 1000, 5000, 30000 };
  unsigned short                  referenceI0 = 0;
  for (const auto workUnits : workUnitCounts)
  {
    // Fresh copy for every run, the filter is in-place and steals the input
    // buffer.
    auto inputCopy = ImageType::New();
    inputCopy->CopyInformation(pristine);
    inputCopy->SetBufferedRegion(pristine->GetBufferedRegion());
    inputCopy->Allocate();
    itk::ImageRegionConstIterator<ImageType> itP(pristine, inputCopy->GetBufferedRegion());
    itk::ImageRegionIterator<ImageType>      itC(inputCopy, inputCopy->GetBufferedRegion());
    for (itP.GoToBegin(), itC.GoToBegin(); !itP.IsAtEnd(); ++itP, ++itC)
    {
      itC.Set(itP.Get());
    }

    auto i0estSweep = rtk::I0EstimationProjectionFilter<ImageType, ImageType, 3>::New();
    i0estSweep->SetInput(inputCopy);
    i0estSweep->SetNumberOfWorkUnits(workUnits);
    i0estSweep->UpdateLargestPossibleRegion();

    const unsigned short i0 = i0estSweep->GetI0();
    if (i0 < minv || i0 > maxv)
    {
      std::cerr << "Test Failed: I0 estimate " << i0 << " out of input range [" << minv << ", " << maxv << "] with "
                << workUnits << " work units" << std::endl;
      return EXIT_FAILURE;
    }
    if (referenceI0 == 0)
    {
      referenceI0 = i0;
    }
    else if (i0 != referenceI0)
    {
      std::cerr << "Test Failed: I0 estimate depends on the number of work units: " << i0 << " with " << workUnits
                << " work units instead of " << referenceI0 << std::endl;
      return EXIT_FAILURE;
    }
  }

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
