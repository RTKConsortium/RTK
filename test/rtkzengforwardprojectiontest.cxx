#include "rtkTest.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkZengForwardProjectionImageFilter.h"
#include <itkImageRegionSplitterDirection.h>
#include <itkImageRegionIterator.h>
#include <cmath>


/**
 * \file rtkzengforwardprojectiontest.cxx
 *
 * \brief Functional test for forward projection
 *
 * The test projects a volume filled with ones. The forward projector should
 * then return the intersection of the ray with the box and it is compared
 * with the analytical intersection of a box with a ray.
 *
 * \author Antoine Robert
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;

#ifdef USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

  using VectorType = itk::Vector<double, 3>;
#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 40;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::PointType   origin;
  ConstantImageSourceType::SizeType    size;
  ConstantImageSourceType::SpacingType spacing;

  // Create Joseph Forward Projector volume input.
  const ConstantImageSourceType::Pointer volInput = ConstantImageSourceType::New();
  origin[0] = -126;
  origin[1] = -126;
  origin[2] = -126;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = 2;
  spacing[0] = 252.;
  spacing[1] = 252.;
  spacing[2] = 252.;
#else
  size[0] = 64;
  size[1] = 64;
  size[2] = 64;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
#endif
  volInput->SetOrigin(origin);
  volInput->SetSpacing(spacing);
  volInput->SetSize(size);
  volInput->SetConstant(1.);
  volInput->UpdateOutputInformation();

  // Initialization Volume, it is used in the Zeng Forward Projector and in the
  // Ray Box Intersection Filter in order to initialize the stack of projections.
  const ConstantImageSourceType::Pointer projInput = ConstantImageSourceType::New();
  size[2] = NumberOfProjectionImages;
  projInput->SetOrigin(origin);
  projInput->SetSpacing(spacing);
  projInput->SetSize(size);
  projInput->SetConstant(0.);
  projInput->Update();

  // Zeng Forward Projection filter
  using JFPType = rtk::ZengForwardProjectionImageFilter<OutputImageType, OutputImageType>;

  JFPType::Pointer jfp = JFPType::New();
  jfp->InPlaceOff();
  jfp->SetInput(projInput->GetOutput());
  jfp->SetInput(1, volInput->GetOutput());

  // Ray Box Intersection filter (reference)
  using RBIType = rtk::RayBoxIntersectionImageFilter<OutputImageType, OutputImageType>;
  RBIType::Pointer rbi = RBIType::New();
  rbi->InPlaceOff();
  rbi->SetInput(projInput->GetOutput());
  VectorType boxMin, boxMax;
  boxMin[0] = -128;
  boxMin[1] = -128;
  boxMin[2] = -128;
  boxMax[0] = 128;
  boxMax[1] = 128;
  boxMax[2] = 128;
  rbi->SetBoxMin(boxMin);
  rbi->SetBoxMax(boxMax);

  std::cout << "\n\n****** Case 1: inner ray source ******" << std::endl;

  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  for (unsigned int i = 0; i < NumberOfProjectionImages; i++)
  {
    geometry->AddProjection(500, 0, i * 360. / NumberOfProjectionImages);
  }

  rbi->SetGeometry(geometry);
  rbi->Update();

  jfp->SetGeometry(geometry);
  jfp->SetAlpha(0.);
  jfp->SetSigmaZero(0.);

  jfp->Update();

  CheckImageQuality<OutputImageType>(jfp->GetOutput(), rbi->GetOutput(), 1.28, 44.0, 255.0);
  std::cout << "\n\nTest PASSED! " << std::endl;


  return EXIT_SUCCESS;
}
