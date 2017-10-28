#include "rtkTest.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkRayCastInterpolatorForwardProjectionImageFilter.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"

/**
 * \file rtkRaycastInterpolatorForwardProjectionTest.cxx
 *
 * \brief Functional test for classes performing Ray Cast Forward projections.
 *
 * This test generates compares the ray casting through a voxelized image (box
 * filled with one or Shepp Logan phantom) and compares the result with
 * analytical calculations. Several geometrical configurations are tested with
 * the source inside or outside the projected volume.
 *
 * \author Simon Rit
 */

int main(int , char** )
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef itk::Vector<double, 3>                   VectorType;
#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfProjectionImages = 3;
#else
  const unsigned int NumberOfProjectionImages = 45;
#endif

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  // The test projects a volume filled with ones. The forward projector should
  // then return the intersection of the ray with the box and it is compared
  // with the analytical intersection of a box with a ray.

  // Create volume input.
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
  volInput->SetOrigin( origin );
  volInput->SetSpacing( spacing );
  volInput->SetSize( size );
  volInput->SetConstant( 1. );
  volInput->UpdateOutputInformation();

  // Initialization Volume, it is used in the forward projector and in the
  // Ray Box Intersection Filter in order to initialize the stack of projections.
  const ConstantImageSourceType::Pointer projInput = ConstantImageSourceType::New();
  size[2] = NumberOfProjectionImages;
  projInput->SetOrigin( origin );
  projInput->SetSpacing( spacing );
  projInput->SetSize( size );
  projInput->SetConstant( 0. );
  projInput->Update();

  // Forward Projection filter
  typedef rtk::RayCastInterpolatorForwardProjectionImageFilter<OutputImageType, OutputImageType> FPType;
  FPType::Pointer fp = FPType::New();
  fp->InPlaceOff();
  fp->SetInput( projInput->GetOutput() );
  fp->SetInput( 1, volInput->GetOutput() );

  // Ray Box Intersection filter (reference)
  typedef rtk::RayBoxIntersectionImageFilter<OutputImageType, OutputImageType> RBIType;
  RBIType::Pointer rbi = RBIType::New();
  rbi->InPlaceOff();
  rbi->SetInput( projInput->GetOutput() );
  VectorType boxMin, boxMax;
  boxMin[0] = -126.0;
  boxMin[1] = -126.0;
  boxMin[2] = -126.0;
  boxMax[0] =  126.0;
  boxMax[1] =  126.0;
  boxMax[2] =  126.0;
  rbi->SetBoxMin(boxMin);
  rbi->SetBoxMax(boxMax);

  std::cout << "\n\n****** Case 1: constant image ******" << std::endl;
  // Geometry
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  for(unsigned int i=0; i<NumberOfProjectionImages; i++)
    geometry->AddProjection(500., 1000., i*8.);

  rbi->SetGeometry( geometry );
  rbi->Update();

  fp->SetGeometry( geometry );
  fp->Update();

  CheckImageQuality<OutputImageType>(rbi->GetOutput(), fp->GetOutput(), 1.25, 43, 255.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Shepp-Logan, outer ray source ******" << std::endl;

  // Create Shepp Logan reference projections
  typedef rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  SLPType::Pointer slp = SLPType::New();
  slp->InPlaceOff();
  slp->SetInput( projInput->GetOutput() );
  slp->SetGeometry(geometry);
  slp->Update();

  // Create a Shepp Logan reference volume (finer resolution)
  origin.Fill(-127);
#if FAST_TESTS_NO_CHECKS
  size.Fill(2);
  spacing.Fill(254.);
#else
  size.Fill(128);
  spacing.Fill(2.);
#endif
  volInput->SetOrigin( origin );
  volInput->SetSpacing( spacing );
  volInput->SetSize( size );
  volInput->SetConstant( 0. );

  typedef rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType> DSLType;
  DSLType::Pointer dsl = DSLType::New();
  dsl->InPlaceOff();
  dsl->SetInput( volInput->GetOutput() );
  dsl->Update();

  // Forward projection
  fp->SetInput( 1, dsl->GetOutput() );
  fp->Update();

  CheckImageQuality<OutputImageType>(slp->GetOutput(), fp->GetOutput(), 1.25, 43, 255.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
