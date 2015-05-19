#include <itkPasteImageFilter.h>
#include <itksys/SystemTools.hxx>

#include "rtkTest.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkCyclicDeformationImageFilter.h"
#include "rtkForwardWarpImageFilter.h"

#include <itkWarpImageFilter.h>

/**
 * \file rtkwarptest.cxx
 *
 * \brief Test for the itkWarpImageFilter and the rtkForwardWarpImageFilter
 *
 * This test generates a phantom, which consists of two
 * ellipsoids, and a Displacement Vector Field (DVF). It warps the phantom
 * backward (using the itkWarpImageFilter and trilinear interpolation) and then
 * forward (using the rtkForwardWarpImageFilter and trilinear splat), and
 * compares the result to the original phantom
 *
 * \author Cyril Mory
 */

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  origin[0] = -63.;
  origin[1] = -31.;
  origin[2] = -63.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 32;
  size[1] = 32;
  size[2] = 32;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
#else
  size[0] = 64;
  size[1] = 32;
  size[2] = 64;
  spacing[0] = 2.;
  spacing[1] = 2.;
  spacing[2] = 2.;
#endif
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );

  // Create vector field
  typedef itk::Vector<float,3>                                                 DVFPixelType;
  typedef itk::Image< DVFPixelType, 3 >                                        DVFImageType;
  typedef itk::ImageRegionIteratorWithIndex< DVFImageType> IteratorType;

  DVFImageType::Pointer deformationField = DVFImageType::New();

  DVFImageType::IndexType startMotion;
  startMotion[0] = 0; // first index on X
  startMotion[1] = 0; // first index on Y
  startMotion[2] = 0; // first index on Z
  DVFImageType::SizeType sizeMotion;
  sizeMotion[0] = 64; // size along X
  sizeMotion[1] = 64; // size along Y
  sizeMotion[2] = 64; // size along Z
  DVFImageType::PointType originMotion;
  originMotion[0] = (sizeMotion[0]-1)*(-0.5); // size along X
  originMotion[1] = (sizeMotion[1]-1)*(-0.5); // size along Y
  originMotion[2] = (sizeMotion[2]-1)*(-0.5); // size along Z
  DVFImageType::RegionType regionMotion;
  regionMotion.SetSize( sizeMotion );
  regionMotion.SetIndex( startMotion );
  deformationField->SetRegions( regionMotion );
  deformationField->SetOrigin(originMotion);
  deformationField->Allocate();

  // Vector Field initilization
  DVFPixelType vec;
  vec.Fill(0.);
  IteratorType defIt( deformationField, deformationField->GetLargestPossibleRegion() );
  for ( defIt.GoToBegin(); !defIt.IsAtEnd(); ++defIt)
    {
      vec.Fill(0.);
      vec[0] = 8.;
      defIt.Set(vec);
    }

  // Create a reference object (in this case a 3D phantom reference).
  // Ellipse 1
  typedef rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType> DEType;
  DEType::Pointer e1 = DEType::New();
  e1->SetInput( tomographySource->GetOutput() );
  e1->SetDensity(2.);
  DEType::VectorType axis;
  axis.Fill(60.);
  e1->SetAxis(axis);
  DEType::VectorType center;
  center.Fill(0.);
  e1->SetCenter(center);
  e1->SetAngle(0.);
  e1->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( e1->Update() )

  // Ellipse 2
  DEType::Pointer e2 = DEType::New();
  e2->SetInput(e1->GetOutput());
  e2->SetDensity(-1.);
  DEType::VectorType axis2;
  axis2.Fill(8.);
  e2->SetAxis(axis2);
  DEType::VectorType center2;
  center2.Fill(0.);
  e2->SetCenter(center2);
  e2->SetAngle(0.);
  e2->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( e2->Update() )

  typedef itk::WarpImageFilter<OutputImageType, OutputImageType, DVFImageType> WarpFilterType;
  WarpFilterType::Pointer warp = WarpFilterType::New();
  warp->SetInput(e2->GetOutput());
  warp->SetDisplacementField( deformationField );
  warp->SetOutputParametersFromImage(e2->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION( warp->Update() );

  typedef rtk::ForwardWarpImageFilter<OutputImageType, OutputImageType, DVFImageType> ForwardWarpFilterType;
  ForwardWarpFilterType::Pointer forwardWarp = ForwardWarpFilterType::New();
  forwardWarp->SetInput(warp->GetOutput());
  forwardWarp->SetDisplacementField( deformationField );
  forwardWarp->SetOutputParametersFromImage(warp->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION( forwardWarp->Update() );

  CheckImageQuality<OutputImageType>(forwardWarp->GetOutput(), e2->GetOutput(), 0.1, 12, 2.0);

  std::cout << "Test PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
