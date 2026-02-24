#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkTestConfiguration.h"
#ifdef USE_CUDA
#  include "rtkCudaLagCorrectionImageFilter.h"
#else
#  include "rtkLagCorrectionImageFilter.h"
#endif

#include <vector>

/**
 * \file rtklagcorrectiontest.cxx
 *
 * \brief Test the lag correction filter
 *
 * \author Sebastien Brousmiche
 */

const unsigned ModelOrder = 4;
const unsigned Nprojections = 10;

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;

  using VectorType = itk::Vector<float, ModelOrder>; // Parameter type always float/double
#ifdef USE_CUDA
  using PixelType = unsigned short;
  using ImageType = itk::CudaImage<PixelType, Dimension>;
  using LCImageFilterType = rtk::CudaLagCorrectionImageFilter;
#else
  using PixelType = float;
  using ImageType = itk::Image<PixelType, Dimension>;
  using LCImageFilterType = rtk::LagCorrectionImageFilter<ImageType, ModelOrder>;
#endif

  auto lagcorr = LCImageFilterType::New();

  auto size = itk::MakeSize(650, 700, 1);

  ImageType::RegionType region;
  region.SetSize(size);

  VectorType a;
  a[0] = 0.7055F;
  a[1] = 0.1141F;
  a[2] = 0.0212F;
  a[3] = 0.0033F;

  VectorType b;
  b[0] = 2.911e-3F;
  b[1] = 0.4454e-3F;
  b[2] = 0.0748e-3F;
  b[3] = 0.0042e-3F;

  lagcorr->SetCoefficients(a, b);

  for (unsigned i = 0; i < Nprojections; ++i)
  {
    auto inputI = ImageType::New();
    inputI->SetRegions(region);
    inputI->Allocate();
    inputI->FillBuffer(1.0F);

    lagcorr->SetInput(inputI.GetPointer());

    TRY_AND_EXIT_ON_ITK_EXCEPTION(lagcorr->Update())
  }

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
