#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkCudaDisplacedDetectorImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkSheppLoganPhantomFilter.h"

#include <itkStreamingImageFilter.h>
#include <itkImageRegionSplitterDirection.h>

/**
 * \file rtkdisplaceddetectorcompcudatest.cxx
 *
 * \brief Test rtk::CudaDisplacedDetectorImageFilter vs rtk::DisplacedDetectorImageFilter
 *
 * This test compares weighted projections using CPU and Cuda implementations
 * of the filter for displaced detector handling in FBP.
 *
 * \author Simon Rit
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::Pointer projSource = ConstantImageSourceType::New();
  projSource->SetOrigin(itk::MakePoint(-127., -3., 0.));
  projSource->SetSpacing(itk::MakeVector(2., 2., 2.));
  projSource->SetSize(itk::MakeSize(128, 4, 4));

  // Geometry
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  geometry->AddProjection(600., 700., 0., 84., 35, 23, 15, 21, 26);
  geometry->AddProjection(500., 800., 45., 21., 12, 16, 546, 14, 41);
  geometry->AddProjection(700., 900., 90., 68., 68, 54, 38, 35, 56);
  geometry->AddProjection(900., 1000., 135., 48., 35, 84, 10, 84, 59);

  // Projections
  using SLPType = rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType>;
  SLPType::Pointer slp = SLPType::New();
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

    using CUDADDFType = rtk::CudaDisplacedDetectorImageFilter;
    CUDADDFType::Pointer cudaddf = CUDADDFType::New();
    cudaddf->SetInput(slp->GetOutput());
    cudaddf->SetGeometry(geometry);
    cudaddf->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION(cudaddf->Update());

    using CPUDDFType = rtk::DisplacedDetectorImageFilter<OutputImageType>;
    CPUDDFType::Pointer cpuddf = CPUDDFType::New();
    cpuddf->SetInput(slp->GetOutput());
    cpuddf->SetGeometry(geometry);
    cpuddf->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION(cpuddf->Update());

    CheckImageQuality<OutputImageType>(cudaddf->GetOutput(), cpuddf->GetOutput(), 1.e-5, 100, 1.);

    std::cout << "\n\n****** Case " << inPlace * 2 + 1 << ": with streaming, ";
    if (!inPlace)
      std::cout << "not";
    std::cout << " in place ******" << std::endl;

    // Idem with streaming
    cudaddf = CUDADDFType::New();
    cudaddf->SetInput(slp->GetOutput());
    cudaddf->SetGeometry(geometry);
    cudaddf->InPlaceOff();

    using StreamingType = itk::StreamingImageFilter<OutputImageType, OutputImageType>;
    StreamingType::Pointer streamingCUDA = StreamingType::New();
    streamingCUDA->SetInput(cudaddf->GetOutput());
    streamingCUDA->SetNumberOfStreamDivisions(4);
    itk::ImageRegionSplitterDirection::Pointer splitter = itk::ImageRegionSplitterDirection::New();
    splitter->SetDirection(2); // Splitting along direction 1, NOT 2
    streamingCUDA->SetRegionSplitter(splitter);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(streamingCUDA->Update());

    cpuddf = CPUDDFType::New();
    cpuddf->SetInput(slp->GetOutput());
    cpuddf->SetGeometry(geometry);
    cpuddf->InPlaceOff();

    StreamingType::Pointer streamingCPU = StreamingType::New();
    streamingCPU->SetInput(cpuddf->GetOutput());
    streamingCPU->SetNumberOfStreamDivisions(2);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(streamingCPU->Update());

    CheckImageQuality<OutputImageType>(streamingCUDA->GetOutput(), streamingCPU->GetOutput(), 1.e-5, 100, 1.);
  }

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
