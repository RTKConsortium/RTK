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

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef float OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;


  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer projSource  = ConstantImageSourceType::New();
  origin[0] = -508.;
  origin[1] = -3.;
  origin[2] = 0.;
  size[0] = 128;
  size[1] = 4;
  size[2] = 4;
  spacing[0] = 8.;
  spacing[1] = 2.;
  spacing[2] = 2.;

  projSource->SetOrigin( origin );
  projSource->SetSpacing( spacing );
  projSource->SetSize( size );

  // Geometry
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  geometry->AddProjection(600., 560.,   0., 3., 0., 0., 0., 2., 0.);
  geometry->AddProjection(500., 545.,  90., 2., 0., 0., 0., 4., 0.);
  geometry->AddProjection(700., 790., 180., 8., 0., 0., 0., 5., 0.);
  geometry->AddProjection(900., 935., 270., 4., 0., 0., 0., 8., 0.);

  // Projections
  typedef rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  SLPType::Pointer slp=SLPType::New();
  slp->SetInput( projSource->GetOutput() );
  slp->SetGeometry(geometry);
  slp->SetPhantomScale(116);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( slp->Update() );

  for(int inPlace=0; inPlace<2; inPlace++)
    {
    std::cout << "\n\n****** Case " << inPlace*2 << ": no streaming, ";
    if(!inPlace)
      std::cout << "not";
    std::cout << " in place ******" << std::endl;

    typedef rtk::DisplacedDetectorForOffsetFieldOfViewImageFilter<OutputImageType> OffsetDDFType;
    OffsetDDFType::Pointer cudaddf = OffsetDDFType::New();
    cudaddf->SetInput( slp->GetOutput() );
    cudaddf->SetGeometry(geometry);
    cudaddf->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION( cudaddf->Update() );

    typedef rtk::DisplacedDetectorImageFilter<OutputImageType> CPUDDFType;
    CPUDDFType::Pointer cpuddf = CPUDDFType::New();
    cpuddf->SetInput( slp->GetOutput() );
    cpuddf->SetGeometry(geometry);
    cpuddf->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION( cpuddf->Update() );

    CheckImageQuality< OutputImageType >(cudaddf->GetOutput(), cpuddf->GetOutput(), 1.e-6, 100, 1.);

    std::cout << "\n\n****** Case " << inPlace*2+1 << ": with streaming, ";
    if(!inPlace)
      std::cout << "not";
    std::cout << " in place ******" << std::endl;

    // Idem with streaming
    cudaddf = OffsetDDFType::New();
    cudaddf->SetInput( slp->GetOutput() );
    cudaddf->SetGeometry(geometry);
    cudaddf->InPlaceOff();

    typedef itk::StreamingImageFilter<OutputImageType, OutputImageType> StreamingType;
    StreamingType::Pointer streamingCUDA = StreamingType::New();
    streamingCUDA->SetInput( cudaddf->GetOutput() );
    streamingCUDA->SetNumberOfStreamDivisions(2);
    TRY_AND_EXIT_ON_ITK_EXCEPTION( streamingCUDA->Update() );

    cpuddf = CPUDDFType::New();
    cpuddf->SetInput( slp->GetOutput() );
    cpuddf->SetGeometry(geometry);
    cpuddf->InPlaceOff();

    StreamingType::Pointer streamingCPU = StreamingType::New();
    streamingCPU->SetInput( cpuddf->GetOutput() );
    streamingCPU->SetNumberOfStreamDivisions(2);
    TRY_AND_EXIT_ON_ITK_EXCEPTION( streamingCPU->Update() );

    CheckImageQuality< OutputImageType >(streamingCUDA->GetOutput(), streamingCPU->GetOutput(), 1.e-6, 100, 1.);
    }

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
