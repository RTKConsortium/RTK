#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkDisplacedDetectorImageFilter.h"
#include "rtkDisplacedDetectorForOffsetFieldOfViewImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkSheppLoganPhantomFilter.h"

#include <itkStreamingImageFilter.h>

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
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::Pointer projSource = ConstantImageSourceType::New();
  projSource->SetOrigin(itk::MakePoint(-508., -3., 0.));
  projSource->SetSpacing(itk::MakeVector(8., 2., 2.));
  projSource->SetSize(itk::MakeSize(128, 4, 4));

  // Geometry
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  geometry->AddProjection(600., 560., 0., 3., 0., 0., 0., 2., 0.);
  geometry->AddProjection(500., 545., 90., 2., 0., 0., 0., 4., 0.);
  geometry->AddProjection(700., 790., 180., 8., 0., 0., 0., 5., 0.);
  geometry->AddProjection(900., 935., 270., 4., 0., 0., 0., 8., 0.);

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

    using OffsetDDFType = rtk::DisplacedDetectorForOffsetFieldOfViewImageFilter<OutputImageType>;
    OffsetDDFType::Pointer cudaddf = OffsetDDFType::New();
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

    CheckImageQuality<OutputImageType>(cudaddf->GetOutput(), cpuddf->GetOutput(), 1.e-6, 100, 1.);

    std::cout << "\n\n****** Case " << inPlace * 2 + 1 << ": with streaming, ";
    if (!inPlace)
      std::cout << "not";
    std::cout << " in place ******" << std::endl;

    // Idem with streaming
    cudaddf = OffsetDDFType::New();
    cudaddf->SetInput(slp->GetOutput());
    cudaddf->SetGeometry(geometry);
    cudaddf->InPlaceOff();

    using StreamingType = itk::StreamingImageFilter<OutputImageType, OutputImageType>;
    StreamingType::Pointer streamingCUDA = StreamingType::New();
    streamingCUDA->SetInput(cudaddf->GetOutput());
    streamingCUDA->SetNumberOfStreamDivisions(2);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(streamingCUDA->Update());

    cpuddf = CPUDDFType::New();
    cpuddf->SetInput(slp->GetOutput());
    cpuddf->SetGeometry(geometry);
    cpuddf->InPlaceOff();

    StreamingType::Pointer streamingCPU = StreamingType::New();
    streamingCPU->SetInput(cpuddf->GetOutput());
    streamingCPU->SetNumberOfStreamDivisions(2);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(streamingCPU->Update());

    CheckImageQuality<OutputImageType>(streamingCUDA->GetOutput(), streamingCPU->GetOutput(), 1.e-6, 100, 1.);
  }

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
