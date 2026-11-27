#include "rtkTest.h"
#include "rtkConstantImageSource.h"
#include "rtkZengForwardProjectionImageFilter.h"
#include "rtkZengBackProjectionImageFilter.h"
#ifdef USE_CUDA
#  include "rtkCudaZengForwardProjectionImageFilter.h"
#  include "rtkCudaZengBackProjectionImageFilter.h"
#endif
#include <itkImageRegionIteratorWithIndex.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace
{
#ifdef USE_CUDA
using ImageType = itk::CudaImage<float, 3>;
using ForwardType = rtk::CudaZengForwardProjectionImageFilter;
using BackType = rtk::CudaZengBackProjectionImageFilter;
#else
using ImageType = itk::Image<float, 3>;
using ForwardType = rtk::ZengForwardProjectionImageFilter<ImageType, ImageType>;
using BackType = rtk::ZengBackProjectionImageFilter<ImageType, ImageType>;
#endif

ImageType::Pointer
MakeImage(const ImageType::SizeType &    size,
          const ImageType::PointType &   origin,
          const ImageType::SpacingType & spacing,
          int                            firstProjection = 0)
{
  auto                 image = ImageType::New();
  ImageType::IndexType index{};
  index[2] = firstProjection;
  image->SetRegions(ImageType::RegionType(index, size));
  image->SetOrigin(origin);
  image->SetSpacing(spacing);
  image->AllocateInitialized();
  return image;
}

ImageType::Pointer
Extend(const ImageType * image, unsigned int width)
{
  auto       size = image->GetLargestPossibleRegion().GetSize();
  const auto padding = (width - size[0]) / 2;
  size[0] = width;
  auto origin = image->GetOrigin();
  for (unsigned int d = 0; d < 3; ++d)
    origin[d] -= padding * image->GetSpacing()[0] * image->GetDirection()[d][0];
  auto wide = MakeImage(size, origin, image->GetSpacing(), image->GetLargestPossibleRegion().GetIndex(2));
  wide->SetDirection(image->GetDirection());
  for (itk::ImageRegionConstIteratorWithIndex<ImageType> it(image, image->GetLargestPossibleRegion()); !it.IsAtEnd();
       ++it)
  {
    auto index = it.GetIndex();
    index[0] += padding;
    wide->SetPixel(index, it.Get());
    ImageType::PointType physical, virtualPoint;
    image->TransformIndexToPhysicalPoint(it.GetIndex(), physical);
    wide->TransformIndexToPhysicalPoint(index, virtualPoint);
    if (physical.EuclideanDistanceTo(virtualPoint) > 1.e-10)
      itkGenericExceptionMacro(<< "Physical detector pixel coordinates changed.");
  }
  return wide;
}

double
Compare(const ImageType * actual, const ImageType * expected, unsigned int padding = 0)
{
  double error = 0., norm = 0.;
  for (itk::ImageRegionConstIteratorWithIndex<ImageType> it(actual, actual->GetBufferedRegion()); !it.IsAtEnd(); ++it)
  {
    auto index = it.GetIndex();
    index[0] += padding;
    const double reference = expected->GetPixel(index);
    error += std::pow(it.Get() - reference, 2);
    norm += reference * reference;
  }
  const double relative = std::sqrt(error / std::max(norm, 1.e-20));
  if (!std::isfinite(relative) || relative > 2.e-5)
    itkGenericExceptionMacro(<< "Virtual detector/reference relative L2 error: " << relative);
  return relative;
}

void
CheckInformation(const ImageType * actual, const ImageType * expected)
{
  if (actual->GetLargestPossibleRegion() != expected->GetLargestPossibleRegion() ||
      actual->GetBufferedRegion() != expected->GetBufferedRegion() ||
      actual->GetRequestedRegion() != expected->GetRequestedRegion() || actual->GetOrigin() != expected->GetOrigin() ||
      actual->GetSpacing() != expected->GetSpacing() || actual->GetDirection() != expected->GetDirection())
    itkGenericExceptionMacro(<< "Virtual detector changed externally visible image information.");
}

void
RequireRejection(itk::ProcessObject * filter)
{
  try
  {
    filter->Update();
  }
  catch (const itk::ExceptionObject & error)
  {
    if (std::string(error.GetDescription()).find("VirtualDetectorPadding") == std::string::npos)
      throw;
    return;
  }
  itkGenericExceptionMacro(<< "Unsupported virtual detector configuration accepted.");
}

#ifdef USE_CUDA
void
Reindex(ImageType * image, const ImageType::IndexType & index)
{
  auto region = image->GetLargestPossibleRegion();
  auto origin = image->GetOrigin();
  for (unsigned int r = 0; r < 3; ++r)
    for (unsigned int c = 0; c < 3; ++c)
      origin[r] += (region.GetIndex(c) - index[c]) * image->GetSpacing()[c] * image->GetDirection()[r][c];
  region.SetIndex(index);
  image->SetRegions(region);
  image->SetOrigin(origin);
}
#endif

double
Dot(const ImageType * a, const ImageType * b, unsigned int padding = 0)
{
  double result = 0.;
  for (itk::ImageRegionConstIteratorWithIndex<ImageType> it(a, a->GetLargestPossibleRegion()); !it.IsAtEnd(); ++it)
  {
    auto index = it.GetIndex();
    index[0] += padding;
    result += static_cast<double>(it.Get()) * b->GetPixel(index);
  }
  return result;
}
} // namespace

int
rtkzengvirtualdetectortest(int, char *[])
{
  try
  {
#ifndef USE_CUDA
    // A connected source may buffer only the region requested by the projector.
    auto source = rtk::ConstantImageSource<ImageType>::New();
    source->SetSize(itk::MakeSize(64, 64, 1));
    source->SetOrigin(itk::MakePoint(-31.5, -31.5, 0.));
    source->SetSpacing(itk::MakeVector(1., 1., 1.));
    source->SetConstant(1.);
    auto smallVolume = MakeImage(itk::MakeSize(8, 8, 8), itk::MakePoint(-3.5, -3.5, -3.5), itk::MakeVector(1., 1., 1.));
    auto smallGeometry = rtk::ThreeDCircularProjectionGeometry::New();
    smallGeometry->AddProjection(65., 0., 0.);
    auto connectedBack = BackType::New();
    connectedBack->InPlaceOff();
    connectedBack->SetInput(0, smallVolume);
    connectedBack->SetInput(1, source->GetOutput());
    connectedBack->SetGeometry(smallGeometry);
    connectedBack->SetVirtualDetectorPadding(4);
    connectedBack->UpdateOutputInformation();
    connectedBack->GetOutput()->SetRequestedRegionToLargestPossibleRegion();
    connectedBack->GetOutput()->PropagateRequestedRegion();
    const auto requested = source->GetOutput()->GetRequestedRegion();
    const auto largest = source->GetOutput()->GetLargestPossibleRegion();
    if (requested.GetIndex(0) != largest.GetIndex(0) || requested.GetSize(0) != largest.GetSize(0) ||
        requested.GetIndex(1) != largest.GetIndex(1) || requested.GetSize(1) != largest.GetSize(1))
      itkGenericExceptionMacro(<< "Padded backprojection did not request the complete detector slice.");
    connectedBack->Update();

    auto fullProjection =
      MakeImage(largest.GetSize(), source->GetOutput()->GetOrigin(), source->GetOutput()->GetSpacing());
    fullProjection->FillBuffer(1.);
    auto referenceBack = BackType::New();
    referenceBack->InPlaceOff();
    referenceBack->SetInput(0, smallVolume);
    referenceBack->SetInput(1, fullProjection);
    referenceBack->SetGeometry(smallGeometry);
    referenceBack->SetVirtualDetectorPadding(4);
    referenceBack->Update();
    Compare(connectedBack->GetOutput(), referenceBack->GetOutput());
#endif
    auto volume =
      MakeImage(itk::MakeSize(24, 8, 16), itk::MakePoint(-28.29, -9.1, -18.45), itk::MakeVector(2.46, 2.6, 2.46));
    auto additiveVolume =
      MakeImage(volume->GetLargestPossibleRegion().GetSize(), volume->GetOrigin(), volume->GetSpacing());
    auto zeroVolume =
      MakeImage(volume->GetLargestPossibleRegion().GetSize(), volume->GetOrigin(), volume->GetSpacing());
    auto attenuation =
      MakeImage(volume->GetLargestPossibleRegion().GetSize(), volume->GetOrigin(), volume->GetSpacing());
    for (itk::ImageRegionIteratorWithIndex<ImageType> it(volume, volume->GetLargestPossibleRegion()); !it.IsAtEnd();
         ++it)
    {
      const auto i = it.GetIndex();
      it.Set(0.2f + (i[0] * 3 + i[1] * 5 + i[2] * 7) % 19 / 19.f);
      additiveVolume->SetPixel(i, 0.1f + i[0] / 30.f);
      attenuation->SetPixel(i, 0.002f + (i[0] + i[2]) % 9 * 0.001f);
    }
    auto nativeForward = ForwardType::New();
    auto nativeBack = BackType::New();
    nativeForward->InPlaceOff();
    nativeBack->InPlaceOff();
    for (unsigned int configuration = 0; configuration < 4; ++configuration)
    {
#ifdef USE_CUDA
      if (configuration == 3)
        for (auto * image :
             { volume.GetPointer(), additiveVolume.GetPointer(), zeroVolume.GetPointer(), attenuation.GetPointer() })
          Reindex(image, ImageType::IndexType{ { 3, -2, 4 } });
#endif
      const unsigned int physicalWidth = configuration == 3 ? 12 : 16;
      const unsigned int views = configuration == 3 ? 1 : 3;
      const int          first = configuration == 3 ? 2 : 0;
      auto               geometry = rtk::ThreeDCircularProjectionGeometry::New();
      for (unsigned int i = 0; i < 3; ++i)
        geometry->AddProjection(65. + i * 8, 0., i * 37. + configuration * 4., i - 1.3, 0.7 - i);
      auto projections = MakeImage(itk::MakeSize(physicalWidth, 8, views),
                                   itk::MakePoint(-18.45, -8.7, 1.3),
                                   itk::MakeVector(2.46, 2.7, 1.2),
                                   first);
      auto zeroProjections = MakeImage(
        projections->GetLargestPossibleRegion().GetSize(), projections->GetOrigin(), projections->GetSpacing(), first);
      auto direction = projections->GetDirection();
      if (configuration == 1)
        direction[0][0] = -1.;
      if (configuration == 3)
      {
        direction[0][0] = direction[1][1] = 0.;
        direction[0][1] = -1.;
        direction[1][0] = 1.;
      }
      projections->SetDirection(direction);
      zeroProjections->SetDirection(direction);
      for (itk::ImageRegionIteratorWithIndex<ImageType> it(projections, projections->GetLargestPossibleRegion());
           !it.IsAtEnd();
           ++it)
      {
        const auto i = it.GetIndex();
        it.Set(0.3f + (i[0] * 11 + i[1] * 7 + i[2]) % 23 / 13.f);
      }
      const bool        psf = configuration & 1;
      const ImageType * mu = configuration & 2 ? attenuation.GetPointer() : nullptr;
      for (unsigned int padding : { 0u, 1u, (60u - physicalWidth) / 2, (256u - physicalWidth) / 2, 22u, 0u })
      {
        const unsigned int virtualWidth = physicalWidth + 2 * padding;
        auto               wide = Extend(projections, virtualWidth);
        auto               wideZero = Extend(zeroProjections, virtualWidth);
        auto               referenceForward = ForwardType::New();
        auto               referenceBack = BackType::New();
        referenceForward->InPlaceOff();
        referenceBack->InPlaceOff();
        for (auto * fp : { nativeForward.GetPointer(), referenceForward.GetPointer() })
        {
          fp->SetInput(1, volume);
          fp->SetInput(2, mu);
          fp->SetGeometry(geometry);
          fp->SetSigmaZero(psf ? 1.4 : 0.);
          fp->SetAlpha(psf ? 0.025 : 0.);
          fp->SetSigmaZeroY(psf ? 1.8 : 0.);
          fp->SetAlphaY(psf ? 0.035 : 0.);
        }
        nativeForward->SetVirtualDetectorPadding(padding);
        nativeForward->SetInput(0, projections);
        referenceForward->SetInput(0, wide);
        nativeForward->UpdateLargestPossibleRegion();
        referenceForward->Update();
#ifdef USE_CUDA
        if (!nativeForward->GetOutput()->GetCudaDataManager()->IsCPUBufferDirty())
          itkGenericExceptionMacro(<< "CUDA forward output was copied to the CPU.");
#endif
        const auto fpError = Compare(nativeForward->GetOutput(), referenceForward->GetOutput(), padding);
        CheckInformation(nativeForward->GetOutput(), projections);
        for (auto * bp : { nativeBack.GetPointer(), referenceBack.GetPointer() })
        {
          bp->SetInput(0, additiveVolume);
          bp->SetInput(2, mu);
          bp->SetGeometry(geometry);
          bp->SetSigmaZero(psf ? 1.4 : 0.);
          bp->SetAlpha(psf ? 0.025 : 0.);
          bp->SetSigmaZeroY(psf ? 1.8 : 0.);
          bp->SetAlphaY(psf ? 0.035 : 0.);
        }
        nativeBack->SetVirtualDetectorPadding(padding);
        nativeBack->SetInput(1, projections);
        referenceBack->SetInput(1, wide);
        nativeBack->UpdateLargestPossibleRegion();
        referenceBack->Update();
#ifdef USE_CUDA
        if (!nativeBack->GetOutput()->GetCudaDataManager()->IsCPUBufferDirty())
          itkGenericExceptionMacro(<< "CUDA backprojection output was copied to the CPU.");
#endif
        const auto bpError = Compare(nativeBack->GetOutput(), referenceBack->GetOutput());
        CheckInformation(nativeBack->GetOutput(), additiveVolume);
        auto mask = MakeImage(projections->GetLargestPossibleRegion().GetSize(),
                              projections->GetOrigin(),
                              projections->GetSpacing(),
                              first);
        mask->SetDirection(projections->GetDirection());
        mask->FillBuffer(1.f);
        nativeBack->SetInput(0, zeroVolume);
        referenceBack->SetInput(0, zeroVolume);
        nativeBack->SetInput(1, mask);
        referenceBack->SetInput(1, Extend(mask, virtualWidth));
        nativeBack->UpdateLargestPossibleRegion();
        referenceBack->Update();
        Compare(nativeBack->GetOutput(), referenceBack->GetOutput());
        nativeBack->SetInput(1, projections);
        referenceBack->SetInput(1, wide);
        nativeBack->UpdateLargestPossibleRegion();
        referenceBack->Update();
        nativeForward->SetInput(0, zeroProjections);
        referenceForward->SetInput(0, wideZero);
        nativeForward->UpdateLargestPossibleRegion();
        referenceForward->Update();
        const double lhs = Dot(projections, nativeForward->GetOutput());
        const double rhs = Dot(volume, nativeBack->GetOutput());
        const double wideLhs = Dot(projections, referenceForward->GetOutput(), padding);
        const double wideRhs = Dot(volume, referenceBack->GetOutput());
        const double adjointError = std::abs(lhs - rhs) / std::max(std::abs(lhs), 1.e-20);
        const double referenceAdjointError = std::abs(wideLhs - wideRhs) / std::max(std::abs(wideLhs), 1.e-20);
        if (std::abs(adjointError - referenceAdjointError) > 2.e-5)
          itkGenericExceptionMacro(<< "Virtual detector changed the reference adjoint discrepancy.");
#ifdef USE_CUDA
        for (unsigned int batch : { 1u, 2u, 8u, 0u })
        {
          nativeForward->SetBatchSize(batch);
          nativeBack->SetBatchSize(batch);
          nativeForward->UpdateLargestPossibleRegion();
          nativeBack->UpdateLargestPossibleRegion();
          Compare(nativeForward->GetOutput(), referenceForward->GetOutput(), padding);
          Compare(nativeBack->GetOutput(), referenceBack->GetOutput());
        }
#endif
        std::cout << "configuration=" << configuration << " padding=" << padding << " FP=" << fpError
                  << " BP=" << bpError << " adjoint discrepancy=" << adjointError << std::endl;
      }
      if (configuration == 0)
      {
        auto subsetInput =
          MakeImage(itk::MakeSize(physicalWidth, 8, 2), projections->GetOrigin(), projections->GetSpacing(), 1);
        subsetInput->FillBuffer(0.7f);
        nativeBack->SetInput(1, subsetInput);
        nativeBack->SetVirtualDetectorPadding(22);
        auto referenceBack = BackType::New();
        referenceBack->InPlaceOff();
        referenceBack->SetInput(0, zeroVolume);
        referenceBack->SetInput(1, Extend(subsetInput, 60));
        referenceBack->SetGeometry(geometry);
        referenceBack->SetSigmaZero(0.);
        referenceBack->SetAlpha(0.);
        nativeBack->UpdateLargestPossibleRegion();
        referenceBack->UpdateLargestPossibleRegion();
        Compare(nativeBack->GetOutput(), referenceBack->GetOutput());

        auto source = rtk::ConstantImageSource<ImageType>::New();
        source->SetInformationFromImage(projections);
        source->SetConstant(0.);
        auto wideSource = rtk::ConstantImageSource<ImageType>::New();
        wideSource->SetInformationFromImage(Extend(projections, 60));
        wideSource->SetConstant(0.);
        auto referenceForward = ForwardType::New();
        referenceForward->InPlaceOff();
        referenceForward->SetInput(0, wideSource->GetOutput());
        referenceForward->SetInput(1, volume);
        referenceForward->SetGeometry(geometry);
        referenceForward->SetSigmaZero(0.);
        referenceForward->SetAlpha(0.);
        nativeForward->SetInput(0, source->GetOutput());
        nativeForward->SetVirtualDetectorPadding(22);
        for (auto * fp : { nativeForward.GetPointer(), referenceForward.GetPointer() })
        {
          fp->UpdateOutputInformation();
          auto requested = fp->GetOutput()->GetLargestPossibleRegion();
          requested.SetIndex(2, 1);
          requested.SetSize(2, 2);
          fp->GetOutput()->SetRequestedRegion(requested);
          fp->Update();
        }
        Compare(nativeForward->GetOutput(), referenceForward->GetOutput(), 22);
        nativeForward->SetInput(0, projections);
        nativeBack->SetInput(1, projections);
        nativeForward->UpdateLargestPossibleRegion();
        nativeBack->UpdateLargestPossibleRegion();
      }
      for (unsigned int invalid :
           { static_cast<unsigned int>(std::numeric_limits<int>::max()), std::numeric_limits<unsigned int>::max() })
      {
        nativeForward->SetVirtualDetectorPadding(invalid);
        nativeBack->SetVirtualDetectorPadding(invalid);
        RequireRejection(nativeForward);
        RequireRejection(nativeBack);
      }
      nativeForward->SetVirtualDetectorPadding(22);
      nativeBack->SetVirtualDetectorPadding(22);
      nativeForward->SetInput(0, projections);
    }
#ifdef USE_CUDA
    for (auto * image :
         { volume.GetPointer(), additiveVolume.GetPointer(), zeroVolume.GetPointer(), attenuation.GetPointer() })
      Reindex(image, ImageType::IndexType{});
#endif
    auto point = MakeImage(volume->GetLargestPossibleRegion().GetSize(), volume->GetOrigin(), volume->GetSpacing());
    const ImageType::IndexType outsideIndex = { { 20, 4, 8 } };
    point->SetPixel(outsideIndex, 1.f);
    auto physical =
      MakeImage(itk::MakeSize(16, 8, 1), itk::MakePoint(-18.45, -8.7, 0.), itk::MakeVector(2.46, 2.7, 1.));
    auto geometry = rtk::ThreeDCircularProjectionGeometry::New();
    geometry->AddProjection(65., 0., 0.);
    auto narrow = ForwardType::New();
    auto extended = ForwardType::New();
    for (auto * fp : { narrow.GetPointer(), extended.GetPointer() })
    {
      fp->InPlaceOff();
      fp->SetInput(0, physical);
      fp->SetInput(1, point);
      fp->SetGeometry(geometry);
      fp->SetSigmaZero(1.4);
      fp->SetAlpha(0.025);
    }
    extended->SetVirtualDetectorPadding(22);
    narrow->UpdateLargestPossibleRegion();
    extended->UpdateLargestPossibleRegion();
    const double narrowSignal = Dot(narrow->GetOutput(), narrow->GetOutput());
    const double extendedSignal = Dot(extended->GetOutput(), extended->GetOutput());
    if (narrowSignal > 1.e-10 || extendedSignal < 1.e-5)
      itkGenericExceptionMacro(<< "PSF from the source outside the physical detector was not recovered.");
    extended->SetSigmaZero(0.);
    extended->SetAlpha(0.);
    extended->UpdateLargestPossibleRegion();
    if (Dot(extended->GetOutput(), extended->GetOutput()) > 1.e-10)
      itkGenericExceptionMacro(<< "Source outside the detector contributed without PSF.");
    std::cout << "Outside-panel source: squared signal " << narrowSignal << " -> " << extendedSignal << std::endl;
  }
  catch (const itk::ExceptionObject & error)
  {
    std::cerr << error << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
