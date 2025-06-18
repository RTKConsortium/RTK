#include "rtkTest.h"
#include "rtkTestConfiguration.h"
#include "rtkMacro.h"
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

  LCImageFilterType::Pointer lagcorr = LCImageFilterType::New();

  auto size = itk::MakeSize(650, 700, 1);

  ImageType::RegionType region;
  region.SetSize(size);

  VectorType a;
  a[0] = 0.7055f;
  a[1] = 0.1141f;
  a[2] = 0.0212f;
  a[3] = 0.0033f;

  VectorType b;
  b[0] = 2.911e-3f;
  b[1] = 0.4454e-3f;
  b[2] = 0.0748e-3f;
  b[3] = 0.0042e-3f;

  lagcorr->SetCoefficients(a, b);

  for (unsigned i = 0; i < Nprojections; ++i)
  {
    ImageType::Pointer inputI = ImageType::New();
    inputI->SetRegions(region);
    inputI->Allocate();
    inputI->FillBuffer(1.0f);

    lagcorr->SetInput(inputI.GetPointer());

    TRY_AND_EXIT_ON_ITK_EXCEPTION(lagcorr->Update())
  }

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
