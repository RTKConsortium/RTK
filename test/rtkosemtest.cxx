#include <itkImageRegionConstIterator.h>

#include "rtkTest.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "itkMaskImageFilter.h"

#ifdef RTK_USE_CUDA
#  include "itkCudaImage.h"
#endif
#include "rtkOSEMConeBeamReconstructionFilter.h"

/**
 * \file rtkosemtest.cxx
 *
 * \brief Functional test for OSEM reconstruction
 *
 * This test generates the projections of an ellipsoid and reconstructs the CT
 * image using the OSEM algorithm with different backprojectors (Voxel-Based,
 * Joseph and CUDA Voxel-Based). The generated results are compared to the
 * expected results (analytical calculation).
 *
 * \author Antoine Robert
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;

#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 60;
#endif


  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::PointType   origin;
  ConstantImageSourceType::SizeType    size;
  ConstantImageSourceType::SpacingType spacing;
  constexpr double                     att = 0.0154;

  ConstantImageSourceType::Pointer tomographySource = ConstantImageSourceType::New();
  ConstantImageSourceType::Pointer volumeSource = ConstantImageSourceType::New();
  ConstantImageSourceType::Pointer attenuationInput = ConstantImageSourceType::New();
  origin[0] = -67.;
  origin[1] = -67.;
  origin[2] = -67.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = 2;
  spacing[0] = 252.;
  spacing[1] = 252.;
  spacing[2] = 252.;
#else
  size[0] = 34;
  size[1] = 34;
  size[2] = 34;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);
  volumeSource->SetOrigin(origin);
  volumeSource->SetSpacing(spacing);
  volumeSource->SetSize(size);
  volumeSource->SetConstant(1.);
  attenuationInput->SetOrigin(origin);
  attenuationInput->SetSpacing(spacing);
  attenuationInput->SetSize(size);
  attenuationInput->SetConstant(att);

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -135.;
  origin[1] = -135.;
  origin[2] = -135.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 504.;
  spacing[1] = 504.;
  spacing[2] = 504.;
#else
  size[0] = 34;
  size[1] = 34;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  // Geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);

  // Create ellipsoid PROJECTIONS
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType>;
  REIType::Pointer rei;

  rei = REIType::New();
  REIType::VectorType semiprincipalaxis, center;
  semiprincipalaxis.Fill(60.);
  center.Fill(0.);
  rei->SetAngle(0.);
  rei->SetDensity(1.);
  rei->SetCenter(center);
  rei->SetAxis(semiprincipalaxis);

  rei->SetInput(projectionsSource->GetOutput());
  rei->SetGeometry(geometry);

  // Update
  TRY_AND_EXIT_ON_ITK_EXCEPTION(rei->Update());

  // Create REFERENCE object (3D ellipsoid).
  using DEType = rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType>;
  DEType::Pointer dsl = DEType::New();
  dsl->SetInput(tomographySource->GetOutput());
  dsl->SetAxis(semiprincipalaxis);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dsl->Update());

  // Create attenuation map according to the reference object
  using MaskFilterType = itk::MaskImageFilter<OutputImageType, OutputImageType>;
  MaskFilterType::Pointer maskFilter = MaskFilterType::New();
  maskFilter->SetInput(attenuationInput->GetOutput());
  maskFilter->SetMaskImage(dsl->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(maskFilter->Update());

  // OSEM reconstruction filtering
  using OSEMType = rtk::OSEMConeBeamReconstructionFilter<OutputImageType>;
  OSEMType::Pointer osem = OSEMType::New();
  osem->SetInput(0, volumeSource->GetOutput());
  osem->SetInput(1, rei->GetOutput());
  osem->SetGeometry(geometry);

  std::cout << "\n\n****** Case 1: Voxel-Based Backprojector, ML-EM 10 iterations ******" << std::endl;

  osem->SetNumberOfIterations(10);
  osem->SetBackProjectionFilter(OSEMType::BP_VOXELBASED);
  osem->SetForwardProjectionFilter(OSEMType::FP_JOSEPH);
  osem->SetNumberOfProjectionsPerSubset(NumberOfProjectionImages);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(osem->Update());

  CheckImageQuality<OutputImageType>(osem->GetOutput(), dsl->GetOutput(), 0.047, 25.0, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout
    << "\n\n****** Case 2: Joseph-Based Backprojector, OS-EM with 10 projections per subset and 4 iterations******"
    << std::endl;

  osem->SetNumberOfIterations(3);
  osem->SetBackProjectionFilter(OSEMType::BP_JOSEPH);
  osem->SetForwardProjectionFilter(OSEMType::FP_JOSEPH);
  osem->SetNumberOfProjectionsPerSubset(10);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(osem->Update());

  CheckImageQuality<OutputImageType>(osem->GetOutput(), dsl->GetOutput(), 0.032, 25.0, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout
    << "\n\n****** Case 3: Voxel-Based Backprojector, OS-EM with 10 projections per subset and 3 iterations******"
    << std::endl;

  osem->SetNumberOfIterations(3);
  osem->SetBackProjectionFilter(OSEMType::BP_VOXELBASED);
  osem->SetForwardProjectionFilter(OSEMType::FP_JOSEPH);
  osem->SetNumberOfProjectionsPerSubset(10);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(osem->Update());

  CheckImageQuality<OutputImageType>(osem->GetOutput(), dsl->GetOutput(), 0.032, 25, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 4: CUDA Voxel-Based Backprojector ******" << std::endl;

  osem->SetBackProjectionFilter(OSEMType::BP_CUDAVOXELBASED);
  osem->SetForwardProjectionFilter(OSEMType::FP_CUDARAYCAST);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(osem->Update());

  CheckImageQuality<OutputImageType>(osem->GetOutput(), dsl->GetOutput(), 0.032, 26, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
#endif

  using ImageIterator = itk::ImageRegionIterator<OutputImageType>;
  ImageIterator itRei(rei->GetOutput(), rei->GetOutput()->GetBufferedRegion());

  itRei.GoToBegin();

  while (!itRei.IsAtEnd())
  {
    typename OutputImageType::PixelType RefVal = itRei.Get();
    if (att == 0)
      itRei.Set(RefVal);
    else
      itRei.Set((1 - exp(-RefVal * att)) / (att));
    ++itRei;
  }
  osem->SetInput(1, rei->GetOutput());
  osem->SetInput(2, maskFilter->GetOutput());

  std::cout
    << "\n\n****** Case 5: Joseph Attenuated Backprojector, OS-EM with 10 projections per subset and 3 iterations******"
    << std::endl;

  osem->SetNumberOfIterations(3);
  osem->SetBackProjectionFilter(OSEMType::BP_JOSEPHATTENUATED);
  osem->SetForwardProjectionFilter(OSEMType::FP_JOSEPHATTENUATED);
  osem->SetNumberOfProjectionsPerSubset(10);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(osem->Update());

  CheckImageQuality<OutputImageType>(osem->GetOutput(), dsl->GetOutput(), 0.032, 25.0, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
