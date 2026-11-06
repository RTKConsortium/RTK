#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkTestConfiguration.h"
#include "rtkLagCorrectionImageFilter.h"
#ifdef USE_CUDA
#  include "rtkCudaLagCorrectionImageFilter.h"
#  include <itkCudaImage.h>
#  include <itkImageRegionIteratorWithIndex.h>
#endif

#include <algorithm>
#include <cmath>
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
rtklagcorrectiontest(int, char *[])
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

#ifdef USE_CUDA
  // Regression: the autoregressive state is shared per detector pixel and
  // used to be concurrently read-modified-written by the threads of all the
  // projections of a stack, breaking the recursion when more than one
  // projection was processed in a single update. Compare the CUDA result on
  // a multi-projection stack with the CPU filter on the same data.
  using FloatImageType = itk::Image<float, Dimension>;
  using CpuFilterType = rtk::LagCorrectionImageFilter<FloatImageType, ModelOrder>;

  const auto stackSize = itk::MakeSize(650, 700, Nprojections);

  auto cpuInput = FloatImageType::New();
  cpuInput->SetRegions(stackSize);
  cpuInput->Allocate();
  itk::ImageRegionIteratorWithIndex<FloatImageType> itSrc(cpuInput, cpuInput->GetBufferedRegion());
  for (itSrc.GoToBegin(); !itSrc.IsAtEnd(); ++itSrc)
  {
    const auto idx = itSrc.GetIndex();
    itSrc.Set(std::min(65500.0f, 1000.0f + 50.0f * (float)idx[2] + 0.1f * (float)(idx[0] % 7)));
  }

  auto cpuStackFilter = CpuFilterType::New();
  cpuStackFilter->SetCoefficients(a, b);
  cpuStackFilter->SetInput(cpuInput);
  cpuStackFilter->Update();
  auto cpuStackOutput = FloatImageType::New();
  cpuStackOutput->Graft(cpuStackFilter->GetOutput());
  cpuStackOutput->DisconnectPipeline();

  auto cudaStackInput = itk::CudaImage<unsigned short, Dimension>::New();
  cudaStackInput->SetRegions(stackSize);
  cudaStackInput->Allocate();
  {
    itk::ImageRegionIterator<itk::CudaImage<unsigned short, Dimension>> itC(cudaStackInput,
                                                                            cudaStackInput->GetBufferedRegion());
    itk::ImageRegionIterator<FloatImageType>                            itF(cpuInput, cpuInput->GetBufferedRegion());
    for (itC.GoToBegin(), itF.GoToBegin(); !itC.IsAtEnd(); ++itC, ++itF)
      itC.Set((unsigned short)itF.Get());
  }

  auto cudaStackFilter = rtk::CudaLagCorrectionImageFilter::New();
  cudaStackFilter->SetCoefficients(a, b);
  cudaStackFilter->SetInput(cudaStackInput);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cudaStackFilter->Update())

  itk::ImageRegionIteratorWithIndex<itk::CudaImage<unsigned short, Dimension>> itCuda(
    cudaStackFilter->GetOutput(), cudaStackFilter->GetOutput()->GetBufferedRegion());
  itk::ImageRegionIterator<FloatImageType> itRef(cpuStackOutput, cpuStackOutput->GetBufferedRegion());
  for (itCuda.GoToBegin(), itRef.GoToBegin(); !itCuda.IsAtEnd(); ++itCuda, ++itRef)
  {
    if (std::fabs((float)itCuda.Get() - itRef.Get()) > 2.0)
    {
      std::cerr << "Test Failed: CUDA lag correction deviates from CPU on a multi-projection stack at index "
                << itCuda.GetIndex() << ": " << itCuda.Get() << " vs " << itRef.Get() << std::endl;
      return EXIT_FAILURE;
    }
  }

  // The kernel writes the state once after the whole stack, so a second
  // update must continue the recursion from the saved state. Process another
  // stack on the same filter instances and compare again: a missing or wrong
  // final state write passes the check above and only fails here.
  auto cpuStack2 = FloatImageType::New();
  cpuStack2->SetRegions(stackSize);
  cpuStack2->Allocate();
  itk::ImageRegionIteratorWithIndex<FloatImageType> itSrc2(cpuStack2, cpuStack2->GetBufferedRegion());
  for (itSrc2.GoToBegin(); !itSrc2.IsAtEnd(); ++itSrc2)
  {
    const auto idx = itSrc2.GetIndex();
    itSrc2.Set(std::min(65500.0f, 4000.0f + 30.0f * (float)idx[2] + 0.2f * (float)(idx[0] % 5)));
  }

  auto cpuInput2 = FloatImageType::New();
  cpuInput2->Graft(cpuStack2);
  cpuInput2->DisconnectPipeline();
  cpuStackFilter->SetInput(cpuInput2);
  cpuStackFilter->Update();
  auto cpuOutput2 = FloatImageType::New();
  cpuOutput2->Graft(cpuStackFilter->GetOutput());
  cpuOutput2->DisconnectPipeline();

  auto cudaStackInput2 = itk::CudaImage<unsigned short, Dimension>::New();
  cudaStackInput2->SetRegions(stackSize);
  cudaStackInput2->Allocate();
  {
    itk::ImageRegionIterator<itk::CudaImage<unsigned short, Dimension>> itC2(cudaStackInput2,
                                                                             cudaStackInput2->GetBufferedRegion());
    itk::ImageRegionIterator<FloatImageType>                            itF2(cpuStack2, cpuStack2->GetBufferedRegion());
    for (itC2.GoToBegin(), itF2.GoToBegin(); !itC2.IsAtEnd(); ++itC2, ++itF2)
      itC2.Set((unsigned short)itF2.Get());
  }
  cudaStackFilter->SetInput(cudaStackInput2);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cudaStackFilter->UpdateLargestPossibleRegion())

  itk::ImageRegionIteratorWithIndex<itk::CudaImage<unsigned short, Dimension>> itCuda2(
    cudaStackFilter->GetOutput(), cudaStackFilter->GetOutput()->GetBufferedRegion());
  itk::ImageRegionIterator<FloatImageType> itRef2(cpuOutput2, cpuOutput2->GetBufferedRegion());
  for (itCuda2.GoToBegin(), itRef2.GoToBegin(); !itCuda2.IsAtEnd(); ++itCuda2, ++itRef2)
  {
    if (std::fabs((float)itCuda2.Get() - itRef2.Get()) > 2.0)
    {
      std::cerr << "Test Failed: CUDA lag correction deviates from CPU on the second multi-projection stack at index "
                << itCuda2.GetIndex() << ": " << itCuda2.Get() << " vs " << itRef2.Get() << std::endl;
      return EXIT_FAILURE;
    }
  }
#endif

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
