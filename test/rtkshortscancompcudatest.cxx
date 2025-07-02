#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkCudaParkerShortScanImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkSheppLoganPhantomFilter.h"

#include <itkStreamingImageFilter.h>
#include <itkImageRegionSplitterDirection.h>

/**
 * \file rtkshortscancompcudatest.cxx
 *
 * \brief Test rtk::CudaParkerShortScanImageFilter vs rtk::ParkerShortScanImageFilter
 *
 * This test compares weighted projections using CPU and Cuda implementations
 * of the filter for short scan handling in FBP.
 *
 * \author Simon Rit
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputImageType = itk::CudaImage<float, Dimension>;


  // Constant image sources
  auto projSource = rtk::ConstantImageSource<OutputImageType>::New();
  auto origin = itk::MakePoint(-127., -3., 0.);
  auto size = itk::MakeSize(128, 4, 4);
  auto spacing = itk::MakeVector(2., 2., 2.);

  projSource->SetOrigin(origin);
  projSource->SetSpacing(spacing);
  projSource->SetSize(size);

  // Geometry
  auto geometry = rtk::ThreeDCircularProjectionGeometry::New();
  geometry->AddProjection(600., 700., 0., 84., 35, 23, 15, 21, 26);
  geometry->AddProjection(500., 800., 45., 21., 12, 16, 546, 14, 41);
  geometry->AddProjection(700., 900., 90., 68., 68, 54, 38, 35, 56);
  geometry->AddProjection(900., 1000., 135., 48., 35, 84, 10, 84, 59);

  // Projections
  auto slp = rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType>::New();
  slp->SetInput(projSource->GetOutput());
  slp->SetGeometry(geometry);
  slp->SetPhantomScale(116);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(slp->Update());

  for (int inPlace = 0; inPlace < 2; inPlace++)
  {
    std::cout << "\n\n****** Case " << inPlace * 2 << ": no streaming, ";
    if (!inPlace)
      std::cout << "not";
    std::cout << " in place ******" << std::endl;

    using CUDASSFType = rtk::CudaParkerShortScanImageFilter;
    auto cudassf = CUDASSFType::New();
    cudassf->SetInput(slp->GetOutput());
    cudassf->SetGeometry(geometry);
    cudassf->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION(cudassf->Update());

    using CPUSSFType = rtk::ParkerShortScanImageFilter<OutputImageType>;
    auto cpussf = CPUSSFType::New();
    cpussf->SetInput(slp->GetOutput());
    cpussf->SetGeometry(geometry);
    cpussf->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION(cpussf->Update());

    CheckImageQuality<OutputImageType>(cudassf->GetOutput(), cpussf->GetOutput(), 1.e-5, 100, 1.);

    std::cout << "\n\n****** Case " << inPlace * 2 + 1 << ": with streaming, ";
    if (!inPlace)
      std::cout << "not";
    std::cout << " in place ******" << std::endl;

    // Idem with streaming
    cudassf = CUDASSFType::New();
    cudassf->SetInput(slp->GetOutput());
    cudassf->SetGeometry(geometry);
    cudassf->InPlaceOff();

    using StreamingType = itk::StreamingImageFilter<OutputImageType, OutputImageType>;
    auto streamingCUDA = StreamingType::New();
    streamingCUDA->SetInput(cudassf->GetOutput());
    streamingCUDA->SetNumberOfStreamDivisions(4);
    auto splitter = itk::ImageRegionSplitterDirection::New();
    splitter->SetDirection(2); // Splitting along direction 1, NOT 2
    streamingCUDA->SetRegionSplitter(splitter);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(streamingCUDA->Update());

    cpussf = CPUSSFType::New();
    cpussf->SetInput(slp->GetOutput());
    cpussf->SetGeometry(geometry);
    cpussf->InPlaceOff();

    auto streamingCPU = StreamingType::New();
    streamingCPU->SetInput(cpussf->GetOutput());
    streamingCPU->SetNumberOfStreamDivisions(2);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(streamingCPU->Update());

    CheckImageQuality<OutputImageType>(streamingCUDA->GetOutput(), streamingCPU->GetOutput(), 1.e-5, 100, 1.);
  }

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
