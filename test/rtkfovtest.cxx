#include <itkImageRegionConstIterator.h>
#include <itkBinaryThresholdImageFilter.h>

#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkFieldOfViewImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkBackProjectionImageFilter.h"

/**
 * \file rtkfovtest.cxx
 *
 * \brief Functional test for classes in charge of creating a FOV (Field Of
 * View) mask
 *
 * This test generates a FOV mask that can be used after a reconstruction.
 * The generated results are compared to the expected results, which are
 * created with a threshold in the backprojection images of the volume.
 *
 * \author Simon Rit and Marc Vila
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 180;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;

  // FOV filter Input Volume, it is used as the input to create the fov mask.
  auto fovInput = ConstantImageSourceType::New();
  auto origin = itk::MakePoint(-127., -127., -127.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(2, 2, 2);
  auto spacing = itk::MakeVector(254., 254., 254.);
#else
  auto size = itk::MakeSize(128, 128, 128);
  auto spacing = itk::MakeVector(2., 2., 2.);
#endif


  fovInput->SetOrigin(origin);
  fovInput->SetSpacing(spacing);
  fovInput->SetSize(size);
  fovInput->SetConstant(1.);

  // BP volume
  auto bpInput = ConstantImageSourceType::New();
  bpInput->SetOrigin(origin);
  bpInput->SetSpacing(spacing);
  bpInput->SetSize(size);

  // BackProjection Input Projections, it is used as the input to create the fov mask.
  auto projectionsSource = ConstantImageSourceType::New();
  origin.Fill(-254.);
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(2, 2, NumberOfProjectionImages);
  spacing = itk::MakeVector(508., 508., 508.);
#else
  size = itk::MakeSize(128, 128, NumberOfProjectionImages);
  spacing = itk::MakeVector(4., 4., 4.);
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(1.);

  std::cout << "\n\n****** Case 1: centered detector ******" << std::endl;

  // Geometry
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto geometry = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600, 1200., noProj * 360. / NumberOfProjectionImages);

  // FOV
  using FOVFilterType = rtk::FieldOfViewImageFilter<OutputImageType, OutputImageType>;
  auto fov = FOVFilterType::New();
  fov->SetInput(0, fovInput->GetOutput());
  fov->SetProjectionsStack(projectionsSource->GetOutput());
  fov->SetGeometry(geometry);
  fov->Update();

  // Backprojection reconstruction filter
  using BPType = rtk::BackProjectionImageFilter<OutputImageType, OutputImageType>;
  auto bp = BPType::New();
  bp->SetInput(0, bpInput->GetOutput());
  bp->SetInput(1, projectionsSource->GetOutput());
  bp->SetGeometry(geometry.GetPointer());

  // Thresholded to the number of projections
  using ThresholdType = itk::BinaryThresholdImageFilter<OutputImageType, OutputImageType>;
  auto threshold = ThresholdType::New();
  threshold->SetInput(bp->GetOutput());
  threshold->SetOutsideValue(0.);
  threshold->SetLowerThreshold(NumberOfProjectionImages - 0.01);
  threshold->SetUpperThreshold(NumberOfProjectionImages + 0.01);
  threshold->SetInsideValue(1.);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(threshold->Update());

  CheckImageQuality<OutputImageType>(fov->GetOutput(), threshold->GetOutput(), 0.02, 23.5, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: offset detector ******" << std::endl;

  origin[0] = -54.;
  projectionsSource->SetOrigin(origin);
  size[0] = 78;
  projectionsSource->SetSize(size);
  projectionsSource->UpdateOutputInformation();
  projectionsSource->UpdateLargestPossibleRegion();
  fov->SetDisplacedDetector(true);
  fov->Update();

  CheckImageQuality<OutputImageType>(fov->GetOutput(), threshold->GetOutput(), 0.02, 23.5, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
