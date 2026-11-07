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
#include <utility>
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
  // multi-projection stacks with the CPU filter on the same data.
  using FloatImageType = itk::Image<float, Dimension>;
  using CpuFilterType = rtk::LagCorrectionImageFilter<FloatImageType, ModelOrder>;

  const auto stackSize = itk::MakeSize(650, 700, Nprojections);

  // Create the same stack for the CPU and the CUDA filters. The values are
  // integers to be exactly represented in both pixel types and vary enough
  // along x so that a wrong state indexing changes the result.
  auto makeStacks = [&](float offset, float slope, unsigned int modulo) {
    auto cpuStack = FloatImageType::New();
    cpuStack->SetRegions(stackSize);
    cpuStack->Allocate();
    auto cudaStack = ImageType::New();
    cudaStack->SetRegions(stackSize);
    cudaStack->Allocate();
    itk::ImageRegionIteratorWithIndex<FloatImageType> itF(cpuStack, cpuStack->GetBufferedRegion());
    itk::ImageRegionIterator<ImageType>               itC(cudaStack, cudaStack->GetBufferedRegion());
    for (; !itF.IsAtEnd(); ++itF, ++itC)
    {
      const auto idx = itF.GetIndex();
      const auto v = static_cast<PixelType>(std::min(65500.f, offset + slope * idx[2] + 2000.f * (idx[0] % modulo)));
      itF.Set(v);
      itC.Set(v);
    }
    return std::make_pair(cpuStack, cudaStack);
  };

  // Process the stacks with both filters on the region and compare. The
  // region is requested after the output information has been updated, as a
  // downstream filter would do, because GenerateOutputInformation resets it.
  auto cpuStackFilter = CpuFilterType::New();
  cpuStackFilter->SetCoefficients(a, b);
  auto cudaStackFilter = rtk::CudaLagCorrectionImageFilter::New();
  cudaStackFilter->SetCoefficients(a, b);
  auto compare = [&](const std::pair<FloatImageType::Pointer, ImageType::Pointer> & stacks,
                     const ImageType::RegionType &                                  reqRegion,
                     const char *                                                   description) {
    cpuStackFilter->SetInput(stacks.first);
    cpuStackFilter->UpdateOutputInformation();
    cpuStackFilter->GetOutput()->SetRequestedRegion(reqRegion);
    cpuStackFilter->Update();
    cudaStackFilter->SetInput(stacks.second);
    cudaStackFilter->UpdateOutputInformation();
    cudaStackFilter->GetOutput()->SetRequestedRegion(reqRegion);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(cudaStackFilter->Update())

    itk::ImageRegionConstIteratorWithIndex<ImageType> itCuda(cudaStackFilter->GetOutput(), reqRegion);
    itk::ImageRegionConstIterator<FloatImageType>     itRef(cpuStackFilter->GetOutput(), reqRegion);
    for (; !itCuda.IsAtEnd(); ++itCuda, ++itRef)
    {
      if (std::fabs((float)itCuda.Get() - itRef.Get()) > 1.5)
      {
        std::cerr << "Test Failed: CUDA lag correction deviates from CPU on " << description << " at index "
                  << itCuda.GetIndex() << ": " << itCuda.Get() << " vs " << itRef.Get() << std::endl;
        return false;
      }
    }
    return true;
  };

  ImageType::RegionType stackRegion(stackSize);
  if (!compare(makeStacks(1000.f, 50.f, 7), stackRegion, "a multi-projection stack"))
    return EXIT_FAILURE;

  // The kernel writes the state once after the whole stack, so a second
  // update must continue the recursion from the saved state. A missing or
  // wrong final state write passes the check above and only fails here.
  if (!compare(makeStacks(4000.f, 30.f, 5), stackRegion, "the second multi-projection stack"))
    return EXIT_FAILURE;

  // The state covers the largest possible region, a requested sub-region
  // must read and update the state of its own pixels.
  ImageType::RegionType subRegion(itk::MakeIndex(101, 53, 0), itk::MakeSize(203, 307, Nprojections));
  if (!compare(makeStacks(2000.f, 40.f, 11), subRegion, "a sub-region of the stack"))
    return EXIT_FAILURE;

  // The state saved by the sub-region update is only used by the next
  // update. Process the whole stack again to check the state inside and
  // outside the sub-region.
  if (!compare(makeStacks(3000.f, 20.f, 13), stackRegion, "the stack following a sub-region"))
    return EXIT_FAILURE;

  // Without lag coefficients, the filter must copy its input.
  auto noLagFilter = rtk::CudaLagCorrectionImageFilter::New();
  auto noLagStacks = makeStacks(1000.f, 50.f, 7);
  noLagFilter->SetInput(noLagStacks.second);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(noLagFilter->Update())
  itk::ImageRegionConstIteratorWithIndex<ImageType> itNoLag(noLagFilter->GetOutput(), stackRegion);
  itk::ImageRegionConstIterator<ImageType>          itNoLagIn(noLagStacks.second, stackRegion);
  for (; !itNoLag.IsAtEnd(); ++itNoLag, ++itNoLagIn)
  {
    if (itNoLag.Get() != itNoLagIn.Get())
    {
      std::cerr << "Test Failed: CUDA lag correction without coefficients modifies the input at index "
                << itNoLag.GetIndex() << ": " << itNoLag.Get() << " vs " << itNoLagIn.Get() << std::endl;
      return EXIT_FAILURE;
    }
  }
#endif

  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
