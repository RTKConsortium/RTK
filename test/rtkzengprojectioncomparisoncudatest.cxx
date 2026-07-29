#include "rtkConstantImageSource.h"
#include "rtkCudaZengBackProjectionImageFilter.h"
#include "rtkCudaZengForwardProjectionImageFilter.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkMacro.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkZengBackProjectionImageFilter.h"
#include "rtkZengForwardProjectionImageFilter.h"

#include <cmath>
#include <cstdlib>
#include <itkCudaImage.h>
#include <itkImageRegionConstIterator.h>
#include <iostream>
#include <limits>

namespace
{
template <typename TReferenceImage, typename TResultImage>
double
RelativeL2Error(const TReferenceImage * reference, const TResultImage * result)
{
  itk::ImageRegionConstIterator<TReferenceImage> referenceIterator(reference, reference->GetBufferedRegion());
  itk::ImageRegionConstIterator<TResultImage>    resultIterator(result, result->GetBufferedRegion());
  double                                         squaredError = 0.;
  double                                         squaredReference = 0.;
  for (referenceIterator.GoToBegin(), resultIterator.GoToBegin(); !referenceIterator.IsAtEnd();
       ++referenceIterator, ++resultIterator)
  {
    const double referenceValue = referenceIterator.Get();
    const double resultValue = resultIterator.Get();
    if (!std::isfinite(resultValue))
    {
      std::cerr << "CUDA Zeng produced a non-finite value." << std::endl;
      return std::numeric_limits<double>::infinity();
    }
    const double difference = resultValue - referenceValue;
    squaredError += difference * difference;
    squaredReference += referenceValue * referenceValue;
  }
  return std::sqrt(squaredError / squaredReference);
}

void
CheckError(const char * label, double value, double tolerance)
{
  std::cout << label << " relative L2 error: " << value << std::endl;
  if (!(value < tolerance))
  {
    std::cerr << label << " relative L2 error exceeds " << tolerance << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
} // namespace

int
rtkzengprojectioncomparisoncudatest(int, char *[])
{
  constexpr unsigned int Dimension = 3;
  using CPUImageType = itk::Image<float, Dimension>;
  using CudaImageType = itk::CudaImage<float, Dimension>;
  using CPUConstantSourceType = rtk::ConstantImageSource<CPUImageType>;
  using CudaConstantSourceType = rtk::ConstantImageSource<CudaImageType>;

  auto cpuVolumeSource = CPUConstantSourceType::New();
  auto cudaVolumeSource = CudaConstantSourceType::New();
  cpuVolumeSource->SetOrigin(itk::MakePoint(-30., -30., -30.));
  cpuVolumeSource->SetSpacing(itk::MakeVector(4., 4., 4.));
  cpuVolumeSource->SetSize(itk::MakeSize(16, 16, 16));
  cpuVolumeSource->SetConstant(0.);
  cudaVolumeSource->SetOrigin(itk::MakePoint(-30., -30., -30.));
  cudaVolumeSource->SetSpacing(itk::MakeVector(4., 4., 4.));
  cudaVolumeSource->SetSize(itk::MakeSize(16, 16, 16));
  cudaVolumeSource->SetConstant(0.);

  auto cpuVolume = rtk::DrawEllipsoidImageFilter<CPUImageType, CPUImageType>::New();
  cpuVolume->SetInput(cpuVolumeSource->GetOutput());
  cpuVolume->SetCenter(itk::MakePoint(0., 0., 0.));
  cpuVolume->SetAxis(itk::MakeVector(20., 16., 12.));
  cpuVolume->SetDensity(1.);
  auto cudaVolume = rtk::DrawEllipsoidImageFilter<CudaImageType, CudaImageType>::New();
  cudaVolume->SetInput(cudaVolumeSource->GetOutput());
  cudaVolume->SetCenter(itk::MakePoint(0., 0., 0.));
  cudaVolume->SetAxis(itk::MakeVector(20., 16., 12.));
  cudaVolume->SetDensity(1.);

  auto cpuProjections = CPUConstantSourceType::New();
  cpuProjections->SetOrigin(itk::MakePoint(-30., -30., 0.));
  cpuProjections->SetSpacing(itk::MakeVector(4., 4., 1.));
  cpuProjections->SetSize(itk::MakeSize(16, 16, 4));
  cpuProjections->SetConstant(0.);
  auto cudaProjections = CudaConstantSourceType::New();
  cudaProjections->SetOrigin(itk::MakePoint(-30., -30., 0.));
  cudaProjections->SetSpacing(itk::MakeVector(4., 4., 1.));
  cudaProjections->SetSize(itk::MakeSize(16, 16, 4));
  cudaProjections->SetConstant(0.);

  auto cpuAttenuation = CPUConstantSourceType::New();
  cpuAttenuation->SetOrigin(itk::MakePoint(-30., -30., -30.));
  cpuAttenuation->SetSpacing(itk::MakeVector(4., 4., 4.));
  cpuAttenuation->SetSize(itk::MakeSize(16, 16, 16));
  cpuAttenuation->SetConstant(0.01);
  auto cudaAttenuation = CudaConstantSourceType::New();
  cudaAttenuation->SetOrigin(itk::MakePoint(-30., -30., -30.));
  cudaAttenuation->SetSpacing(itk::MakeVector(4., 4., 4.));
  cudaAttenuation->SetSize(itk::MakeSize(16, 16, 16));
  cudaAttenuation->SetConstant(0.01);

  auto geometry = rtk::ThreeDCircularProjectionGeometry::New();
  for (unsigned int projection = 0; projection < 4; ++projection)
    geometry->AddProjection(100., 0., projection * 90.);

  using CPUForwardType = rtk::ZengForwardProjectionImageFilter<CPUImageType, CPUImageType>;
  auto cpuForward = CPUForwardType::New();
  cpuForward->InPlaceOff();
  cpuForward->SetInput(0, cpuProjections->GetOutput());
  cpuForward->SetInput(1, cpuVolume->GetOutput());
  cpuForward->SetGeometry(geometry);
  cpuForward->SetSigmaZero(1.5);
  cpuForward->SetAlpha(0.016);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cpuForward->Update());

  auto cudaForward = rtk::CudaZengForwardProjectionImageFilter::New();
  cudaForward->InPlaceOff();
  cudaForward->SetInput(0, cudaProjections->GetOutput());
  cudaForward->SetInput(1, cudaVolume->GetOutput());
  cudaForward->SetGeometry(geometry);
  cudaForward->SetSigmaZero(1.5);
  cudaForward->SetAlpha(0.016);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cudaForward->Update());
  if (!cudaForward->GetOutput()->GetCudaDataManager()->IsCPUBufferDirty())
  {
    std::cerr << "CUDA Zeng forward output was synchronized to the CPU." << std::endl;
    return EXIT_FAILURE;
  }
  CheckError("Forward", RelativeL2Error(cpuForward->GetOutput(), cudaForward->GetOutput()), 5.e-5);

  using CPUBackType = rtk::ZengBackProjectionImageFilter<CPUImageType, CPUImageType>;
  auto cpuBack = CPUBackType::New();
  cpuBack->InPlaceOff();
  cpuBack->SetInput(0, cpuVolumeSource->GetOutput());
  cpuBack->SetInput(1, cpuForward->GetOutput());
  cpuBack->SetGeometry(geometry);
  cpuBack->SetSigmaZero(1.5);
  cpuBack->SetAlpha(0.016);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cpuBack->Update());

  auto cudaBack = rtk::CudaZengBackProjectionImageFilter::New();
  cudaBack->InPlaceOff();
  cudaBack->SetInput(0, cudaVolumeSource->GetOutput());
  cudaBack->SetInput(1, cudaForward->GetOutput());
  cudaBack->SetGeometry(geometry);
  cudaBack->SetSigmaZero(1.5);
  cudaBack->SetAlpha(0.016);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cudaBack->Update());
  if (!cudaBack->GetOutput()->GetCudaDataManager()->IsCPUBufferDirty())
  {
    std::cerr << "CUDA Zeng backprojection output was synchronized to the CPU." << std::endl;
    return EXIT_FAILURE;
  }
  CheckError("Backprojection", RelativeL2Error(cpuBack->GetOutput(), cudaBack->GetOutput()), 5.e-5);

  cpuForward->SetInput(2, cpuAttenuation->GetOutput());
  cpuForward->Modified();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cpuForward->Update());
  cudaForward->SetInput(2, cudaAttenuation->GetOutput());
  cudaForward->Modified();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cudaForward->Update());
  if (!cudaForward->GetOutput()->GetCudaDataManager()->IsCPUBufferDirty())
  {
    std::cerr << "Attenuated CUDA Zeng forward output was synchronized to the CPU." << std::endl;
    return EXIT_FAILURE;
  }
  CheckError(
    "Attenuated forward", RelativeL2Error(cpuForward->GetOutput(), cudaForward->GetOutput()), 5.e-4);

  cpuBack->SetInput(1, cpuForward->GetOutput());
  cpuBack->SetInput(2, cpuAttenuation->GetOutput());
  cpuBack->Modified();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cpuBack->Update());
  cudaBack->SetInput(1, cudaForward->GetOutput());
  cudaBack->SetInput(2, cudaAttenuation->GetOutput());
  cudaBack->Modified();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cudaBack->Update());
  if (!cudaBack->GetOutput()->GetCudaDataManager()->IsCPUBufferDirty())
  {
    std::cerr << "Attenuated CUDA Zeng backprojection output was synchronized to the CPU." << std::endl;
    return EXIT_FAILURE;
  }
  CheckError("Attenuated backprojection",
             RelativeL2Error(cpuBack->GetOutput(), cudaBack->GetOutput()),
             2.e-3);

  std::cout << "CUDA Zeng CPU comparison test PASSED." << std::endl;
  return EXIT_SUCCESS;
}
