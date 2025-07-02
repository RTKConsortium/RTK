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

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputImageType = itk::Image<float, Dimension>;

  // Constant image sources
  auto tomographySource = rtk::ConstantImageSource<OutputImageType>::New();
  auto origin = itk::MakePoint(-63., -31., -63.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(32, 32, 32);
  auto spacing = itk::MakeVector(8., 8., 8.);
#else
  auto size = itk::MakeSize(64, 32, 64);
  auto spacing = itk::MakeVector(2., 2., 2.);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  // Create vector field
  using DVFPixelType = itk::Vector<float, 3>;
  using DVFImageType = itk::Image<DVFPixelType, 3>;

  auto deformationField = DVFImageType::New();

  auto                    sizeMotion = itk::MakeSize(64, 64, 64);
  DVFImageType::PointType originMotion;
  originMotion[0] = (sizeMotion[0] - 1) * (-0.5); // size along X
  originMotion[1] = (sizeMotion[1] - 1) * (-0.5); // size along Y
  originMotion[2] = (sizeMotion[2] - 1) * (-0.5); // size along Z
  DVFImageType::RegionType regionMotion;
  regionMotion.SetSize(sizeMotion);
  deformationField->SetRegions(regionMotion);
  deformationField->SetOrigin(originMotion);
  deformationField->Allocate();

  // Vector Field initilization
  DVFPixelType vec;
  vec.Fill(0.);
  itk::ImageRegionIteratorWithIndex<DVFImageType> defIt(deformationField, deformationField->GetLargestPossibleRegion());
  for (defIt.GoToBegin(); !defIt.IsAtEnd(); ++defIt)
  {
    vec.Fill(0.);
    vec[0] = 8.;
    defIt.Set(vec);
  }

  // Create a reference object (in this case a 3D phantom reference).
  // Ellipse 1
  using DEType = rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType>;
  auto e1 = DEType::New();
  e1->SetInput(tomographySource->GetOutput());
  e1->SetDensity(2.);
  DEType::VectorType axis;
  axis.Fill(60.);
  e1->SetAxis(axis);
  DEType::VectorType center;
  center.Fill(0.);
  e1->SetCenter(center);
  e1->SetAngle(0.);
  e1->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(e1->Update())

  // Ellipse 2
  auto e2 = DEType::New();
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
  TRY_AND_EXIT_ON_ITK_EXCEPTION(e2->Update())

  auto warp = itk::WarpImageFilter<OutputImageType, OutputImageType, DVFImageType>::New();
  warp->SetInput(e2->GetOutput());
  warp->SetDisplacementField(deformationField);
  warp->SetOutputParametersFromImage(e2->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION(warp->Update());

  auto forwardWarp = rtk::ForwardWarpImageFilter<OutputImageType, OutputImageType, DVFImageType>::New();
  forwardWarp->SetInput(warp->GetOutput());
  forwardWarp->SetDisplacementField(deformationField);
  forwardWarp->SetOutputParametersFromImage(warp->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION(forwardWarp->Update());

  CheckImageQuality<OutputImageType>(forwardWarp->GetOutput(), e2->GetOutput(), 0.1, 12, 2.0);

  std::cout << "Test PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
