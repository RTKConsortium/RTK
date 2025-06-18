#include "rtkTest.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkConstantImageSource.h"
#include "rtkJosephForwardAttenuatedProjectionImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include <itkStreamingImageFilter.h>
#include <itkImageRegionSplitterDirection.h>
#include <itkSubtractImageFilter.h>
#include <itkCenteredEuler3DTransform.h>
#include <itkPasteImageFilter.h>
#include <cmath>
#include <itkMemoryUsageObserver.h>


#ifdef USE_CUDA
#  include "rtkCudaForwardProjectionImageFilter.h"
#else
#  include "rtkJosephForwardAttenuatedProjectionImageFilter.h"
#endif

/**
 * \file rtkforwardattenuatedprojectiontest.cxx
 *
 * \brief Functional test for forward projection
 *
 * The test projects a volume filled with ones. The forward projector should
 * then return the intersection of the ray with the box and it is compared
 * with the analytical intersection of a box with a ray.
 *
 * \author Simon Rit and Marc Vila
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

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
#else
  constexpr unsigned int NumberOfProjectionImages = 45;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  constexpr double att = 0.0154;

  // Create Joseph Forward Projector volume input.
  const ConstantImageSourceType::Pointer volInput = ConstantImageSourceType::New();
  auto                                   origin = itk::MakePoint(-126., -126., -126.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(2, 2, 2);
  auto spacing = itk::MakeVector(252., 252., 252.);
#else
  auto size = itk::MakeSize(64, 64, 64);
  auto spacing = itk::MakeVector(4., 4., 4.);
#endif
  volInput->SetOrigin(origin);
  volInput->SetSpacing(spacing);
  volInput->SetSize(size);
  volInput->SetConstant(0.);
  volInput->UpdateOutputInformation();

  // Create Joseph Forward Projector attenuation map.
  const ConstantImageSourceType::Pointer attenuationInput = ConstantImageSourceType::New();

  attenuationInput->SetOrigin(origin);
  attenuationInput->SetSpacing(spacing);
  attenuationInput->SetSize(size);
  attenuationInput->SetConstant(0);

  using DEIFType = rtk::DrawEllipsoidImageFilter<OutputImageType, OutputImageType>;
  DEIFType::Pointer deif = DEIFType::New();
  auto              axis_vol = itk::MakeVector(50., 50., 50.);
  auto              center_vol = itk::MakeVector(0., 0., -30.);
  deif->SetInput(volInput->GetOutput());
  deif->SetCenter(center_vol);
  deif->SetAxis(axis_vol);
  deif->SetDensity(1);
  deif->Update();

  typename OutputImageType::Pointer attenuationMap, volumeSource;
  volumeSource = deif->GetOutput();
  volumeSource->DisconnectPipeline();
  deif->SetCenter(itk::MakeVector(0., 0., 0.));
  deif->SetAxis(itk::MakeVector(90, 90, 90));
  deif->SetDensity(att);
  deif->Update();
  attenuationMap = deif->GetOutput();
  attenuationMap->DisconnectPipeline();

  // Initialization Volume, it is used in the Joseph Forward Projector and in the
  // Ray Box Intersection Filter in order to initialize the stack of projections.
  const ConstantImageSourceType::Pointer projInput = ConstantImageSourceType::New();
  size[2] = 1;
  projInput->SetOrigin(origin);
  projInput->SetSpacing(spacing);
  projInput->SetSize(size);
  projInput->SetConstant(0.);
  projInput->Update();

  const ConstantImageSourceType::Pointer projTotal = ConstantImageSourceType::New();
  size[2] = NumberOfProjectionImages;
  projTotal->SetOrigin(origin);
  projTotal->SetSpacing(spacing);
  projTotal->SetSize(size);
  projTotal->SetConstant(0.);
  projTotal->Update();

  // Joseph Forward Projection filter
#ifdef USE_CUDA
  using JFPType = rtk::CudaForwardProjectionImageFilter<OutputImageType, OutputImageType>;
#else
  using JFPType = rtk::JosephForwardAttenuatedProjectionImageFilter<OutputImageType, OutputImageType>;
#endif
  JFPType::Pointer jfp = JFPType::New();
  jfp->InPlaceOff();
  jfp->SetInput(projTotal->GetOutput());
  jfp->SetInput(1, volumeSource);
  jfp->SetInput(2, attenuationMap);

#ifdef USE_CUDA
  jfp->SetStepSize(10);
#endif
  // Geometry
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry_projection = GeometryType::New();
  double                angle = 0;
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType>;
  REIType::PointType  center_transform;
  REIType::VectorType clip_plane_direction_init, clip_plane_direction;
  clip_plane_direction_init[0] = 0.;
  clip_plane_direction_init[1] = 0.;
  clip_plane_direction_init[2] = 1.;
  using TransformType = itk::CenteredEuler3DTransform<double>;
  typename TransformType::Pointer transform = TransformType::New();
  using SubtractImageFilterType = itk::SubtractImageFilter<OutputImageType, OutputImageType>;
  typename SubtractImageFilterType::Pointer subtractImageFilter = SubtractImageFilterType::New();
  typename OutputImageType::IndexType       indexSlice;
  typename OutputImageType::Pointer         pimg;
  indexSlice.Fill(0);
  using PasteImageFilterType = itk::PasteImageFilter<OutputImageType, OutputImageType>;
  typename PasteImageFilterType::Pointer pasteImageFilter = PasteImageFilterType::New();
  pasteImageFilter->SetDestinationImage(projTotal->GetOutput());

  int count = 0;
  for (unsigned int i = 0; i < NumberOfProjectionImages; i++)
  {
    angle = i * 360. / NumberOfProjectionImages;
    geometry_projection->AddProjection(500, 0., angle);
    transform->SetRotation(0., angle * itk::Math::pi / 180, 0.);
    clip_plane_direction = transform->GetMatrix() * clip_plane_direction_init;
    center_transform = transform->GetMatrix() * center_vol;
    REIType::Pointer sphere_attenuation = REIType::New();
    sphere_attenuation->SetAngle(0);
    sphere_attenuation->SetDensity(1);
    sphere_attenuation->SetCenter(itk::MakeVector(0., 0., 0.));
    sphere_attenuation->SetAxis(itk::MakeVector(90, 90, 90));
    sphere_attenuation->SetInput(projInput->GetOutput());
    REIType::Pointer sphere_emission = REIType::New();
    sphere_emission->SetAngle(0);
    sphere_emission->SetDensity(1);
    sphere_emission->SetCenter(center_vol);
    sphere_emission->SetAxis(axis_vol);
    sphere_emission->SetInput(projInput->GetOutput());
    sphere_attenuation->AddClipPlane(clip_plane_direction, center_transform[2]);
    sphere_attenuation->SetGeometry(geometry_projection);
    sphere_emission->AddClipPlane(clip_plane_direction, center_transform[2]);
    sphere_emission->SetGeometry(geometry_projection);
    indexSlice[2] = count;
    sphere_attenuation->Update();
    sphere_emission->Update();
    subtractImageFilter->SetInput1(sphere_attenuation->GetOutput());
    subtractImageFilter->SetInput2(sphere_emission->GetOutput());
    subtractImageFilter->Update();
    pasteImageFilter->SetSourceImage(subtractImageFilter->GetOutput());
    pasteImageFilter->SetSourceRegion(subtractImageFilter->GetOutput()->GetLargestPossibleRegion());
    pasteImageFilter->SetDestinationIndex(indexSlice);
    pasteImageFilter->Update();
    pimg = pasteImageFilter->GetOutput();
    pimg->DisconnectPipeline();
    pasteImageFilter->SetDestinationImage(pimg);
    geometry_projection->Clear();
    count += 1;
  }

  // Streaming filter to test for unusual regions
  using StreamingFilterType = itk::StreamingImageFilter<OutputImageType, OutputImageType>;
  StreamingFilterType::Pointer stream = StreamingFilterType::New();
  stream->SetInput(jfp->GetOutput());

  stream->SetNumberOfStreamDivisions(9);
  itk::ImageRegionSplitterDirection::Pointer splitter = itk::ImageRegionSplitterDirection::New();
  splitter->SetDirection(2); // Splitting along direction 1, NOT 2
  stream->SetRegionSplitter(splitter);

  std::cout << "\n\n****** Case 1: inner ray source ******" << std::endl;

  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  for (unsigned int i = 0; i < NumberOfProjectionImages; i++)
  {
    geometry->AddProjection(500, 0., i * 360. / NumberOfProjectionImages);
  }
  projTotal->Update();
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType>;
  REIType::Pointer rei = REIType::New();
  rei->InPlaceOff();
  rei->SetAngle(0);
  rei->SetDensity(1);
  rei->SetCenter(center_vol);
  rei->SetAxis(axis_vol);
  rei->SetInput(projTotal->GetOutput());
  rei->SetGeometry(geometry);
  rei->Update();

  using CustomBinaryFilterType = itk::BinaryGeneratorImageFilter<OutputImageType, OutputImageType, OutputImageType>;
  typename CustomBinaryFilterType::Pointer customBinaryFilter = CustomBinaryFilterType::New();
  // Set Lambda function
  auto customLambda = [&](const typename OutputImageType::PixelType & input1,
                          const typename OutputImageType::PixelType & input2) -> typename OutputImageType::PixelType {
    return static_cast<typename OutputImageType::PixelType>((1 - std::exp(-input1 * att)) / att *
                                                            std::exp(-input2 * att));
  };
  customBinaryFilter->SetFunctor(customLambda);
  customBinaryFilter->SetInput1(rei->GetOutput());
  customBinaryFilter->SetInput2(pimg);
  customBinaryFilter->Update();
  jfp->SetGeometry(geometry);
  stream->Update();

  CheckImageQuality<OutputImageType>(stream->GetOutput(), customBinaryFilter->GetOutput(), 1.28, 44.0, 255.0);
  std::cout << "\n\nTest  PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
