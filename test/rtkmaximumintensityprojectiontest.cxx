#include "rtkTest.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkJosephForwardProjectionImageFilter.h"
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
  ConstantImageSourceType::PointType   origin;
  ConstantImageSourceType::SizeType    size;
  ConstantImageSourceType::SizeType    sizeOutput;
  ConstantImageSourceType::SpacingType spacing;
  ConstantImageSourceType::SpacingType spacingOutput;

  // Create MIP Forward Projector volume input.
  const ConstantImageSourceType::Pointer volInput = ConstantImageSourceType::New();
  origin[0] = 0;
  origin[1] = 0;
  origin[2] = 0;
  size[0] = 64;
  size[1] = 64;
  size[2] = 64;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;

  sizeOutput[0] = 200;
  sizeOutput[1] = 200;
  sizeOutput[2] = 200;
  spacingOutput[0] = 1.;
  spacingOutput[1] = 1.;
  spacingOutput[2] = 1.;

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

  // Test-1
  // Joseph Forward Projection filter for MIP projection
  std::cout << "\n\n****** Case 1: JosephForwardProjection filter with custom MIP functions ******" << std::endl;
  using JFPType = rtk::JosephForwardProjectionImageFilter<OutputImageType, OutputImageType>;
  JFPType::Pointer jfp = JFPType::New();
  jfp->SetInput(projInput->GetOutput());
  jfp->SetInput(1, volInput->GetOutput());

  // Function to compute the maximum intensity (MIP) value along the ray projection.
  JFPType::SumAlongRayFunc sumAlongFunc = [](const itk::ThreadIdType,
                                             JFPType::OutputPixelType &    mipValue,
                                             const JFPType::InputPixelType volumeValue,
                                             const JFPType::VectorType &) -> void {
    JFPType::OutputPixelType tmp = static_cast<JFPType::OutputPixelType>(volumeValue);
    if (tmp > mipValue)
    {
      mipValue = tmp;
    }
  };
  jfp->SetSumAlongRay(sumAlongFunc);

  // Performs a MIP forward projection, i.e. calculation of a maximum intensity
  // step along the x-ray line.
  JFPType::ProjectedValueAccumulationFunc projAccumFunc = [](const itk::ThreadIdType          itkNotUsed(threadId),
                                                             const JFPType::InputPixelType &  input,
                                                             JFPType::OutputPixelType &       output,
                                                             const JFPType::OutputPixelType & rayCastValue,
                                                             const JFPType::VectorType &      stepInMM,
                                                             const JFPType::VectorType &      itkNotUsed(source),
                                                             const JFPType::VectorType &      itkNotUsed(sourceToPixel),
                                                             const JFPType::VectorType &      itkNotUsed(nearestPoint),
                                                             const JFPType::VectorType & itkNotUsed(farthestPoint)) {
    JFPType::OutputPixelType tmp = static_cast<JFPType::OutputPixelType>(input);
    if (tmp < rayCastValue)
    {
      tmp = rayCastValue;
    }
    output = tmp * stepInMM.GetNorm();
  };
  jfp->SetProjectedValueAccumulation(projAccumFunc);

  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  geometry->AddProjection(700, 800, 0);

  jfp->SetGeometry(geometry);
  jfp->Update();

  using ConstIteratorType = itk::ImageRegionConstIterator<OutputImageType>;
  ConstIteratorType inputIt(jfp->GetOutput(), jfp->GetOutput()->GetRequestedRegion());

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

  // Test-2
  // Maximum Intensity Projection (MIP) filter, derived from Joseph Forward Projection filter
  std::cout << "\n\n****** Case 2: MaximumIntensityProjection (MIP) filter ******" << std::endl;
  using MIPType = rtk::MaximumIntensityProjectionImageFilter<OutputImageType, OutputImageType>;
  MIPType::Pointer mipfp = MIPType::New();
  mipfp->SetInput(projInput->GetOutput());
  mipfp->SetInput(1, volInput->GetOutput());

  mipfp->SetGeometry(geometry);
  mipfp->Update();

  inputIt = ConstIteratorType(mipfp->GetOutput(), mipfp->GetOutput()->GetRequestedRegion());

  inputIt.GoToBegin();

  res = false;
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

  std::cout << "\n\n****** Compare two resulted images ******" << std::endl;
  CheckImageQuality<OutputImageType>(jfp->GetOutput(), mipfp->GetOutput(), 0.001, 0.000001, 1.E+19);
  std::cout << "\n\nImages are OK! " << std::endl;

  return EXIT_SUCCESS;
}
