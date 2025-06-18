#include "rtkTest.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkMaximumIntensityProjectionImageFilter.h"
#include <itkImageRegionSplitterDirection.h>
#include <itkImageRegionIterator.h>
#include <itkImageFileWriter.h>
#include <itkShiftScaleImageFilter.h>
#include <cmath>


/**
 * \file rtkmaximumintensityprojectiontest.cxx
 *
 * \brief Functional test for MIP forward filter
 *
 * The test projects a volume filled with ones.
 *
 * \author Mikhail Polkovnikov
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  constexpr unsigned int NumberOfProjectionImages = 1;

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  // Create MIP Forward Projector volume input.
  const ConstantImageSourceType::Pointer volInput = ConstantImageSourceType::New();

  auto origin = itk::MakePoint(0., 0., 0.);
  auto size = itk::MakeSize(64, 64, 64);
  auto spacing = itk::MakeVector(4., 4., 4.);
  auto sizeOutput = itk::MakeSize(200, 200, 200);
  auto spacingOutput = itk::MakeVector(1., 1., 1.);

  volInput->SetOrigin(origin);
  volInput->SetSpacing(spacing);
  volInput->SetSize(size);
  volInput->SetConstant(1.);
  volInput->UpdateOutputInformation();

  // Initialization Imager Volume
  const ConstantImageSourceType::Pointer projInput = ConstantImageSourceType::New();
  sizeOutput[2] = NumberOfProjectionImages;
  projInput->SetOrigin(origin);
  projInput->SetSpacing(spacingOutput);
  projInput->SetSize(sizeOutput);
  projInput->SetConstant(0.);
  projInput->Update();

  // MIP Forward Projection filter
  using MIPType = rtk::MaximumIntensityProjectionImageFilter<OutputImageType, OutputImageType>;
  MIPType::Pointer mipfp = MIPType::New();
  mipfp->SetInput(projInput->GetOutput());
  mipfp->SetInput(1, volInput->GetOutput());

  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  geometry->AddProjection(700, 800, 0);

  mipfp->SetGeometry(geometry);
  mipfp->Update();

  using ConstIteratorType = itk::ImageRegionConstIterator<OutputImageType>;
  ConstIteratorType inputIt(mipfp->GetOutput(), mipfp->GetOutput()->GetRequestedRegion());

  inputIt.GoToBegin();

  bool res = false;
  while (!inputIt.IsAtEnd())
  {
    OutputPixelType pixel = inputIt.Get();
    if (pixel < 4. || pixel > 4.25)
    {
      res = true;
    }
    ++inputIt;
  }
  if (res)
  {
    return EXIT_FAILURE;
  }
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
