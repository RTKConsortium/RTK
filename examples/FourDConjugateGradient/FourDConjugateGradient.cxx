// #include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkIterationCommands.h"
#include "rtkSignalToInterpolationWeights.h"
#include "rtkReorderProjectionsImageFilter.h"
#include "../../test/rtkFourDTestHelper.h"

#ifdef RTK_USE_CUDA
#  include <itkCudaImage.h>
#endif
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  using OutputPixelType = float;

#ifdef RTK_USE_CUDA
  using VolumeSeriesType = itk::CudaImage<OutputPixelType, 4>;
  using ProjectionStackType = itk::CudaImage<OutputPixelType, 3>;
  using VolumeType = itk::CudaImage<OutputPixelType, 3>;
#else
  using VolumeSeriesType = itk::Image<OutputPixelType, 4>;
  using ProjectionStackType = itk::Image<OutputPixelType, 3>;
  using VolumeType = itk::Image<OutputPixelType, 3>;
#endif

  auto data = rtk::GenerateFourDTestData<OutputPixelType>(FAST_TESTS_NO_CHECKS);

  // Re-order geometry and projections
  // In the new order, projections with identical phases are packed together
  auto reorder = rtk::ReorderProjectionsImageFilter<ProjectionStackType>::New();
  reorder->SetInput(data.Projections);
  reorder->SetInputGeometry(data.Geometry);
  reorder->SetInputSignal(data.Signal);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reorder->Update())

  // Release the memory holding the stack of original projections
  data.Projections->ReleaseData();

  // Compute the interpolation weights
  auto signalToInterpolationWeights = rtk::SignalToInterpolationWeights::New();
  signalToInterpolationWeights->SetSignal(reorder->GetOutputSignal());
  signalToInterpolationWeights->SetNumberOfReconstructedFrames(
    data.InitialVolumeSeries->GetLargestPossibleRegion().GetSize(3));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(signalToInterpolationWeights->Update())

  // Set the forward and back projection filters to be used
  using ConjugateGradientFilterType =
    rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>;
  auto fourdconjugategradient = ConjugateGradientFilterType::New();

  fourdconjugategradient->SetInputVolumeSeries(data.InitialVolumeSeries);
  fourdconjugategradient->SetNumberOfIterations(2);
#ifdef RTK_USE_CUDA
  fourdconjugategradient->SetCudaConjugateGradient(true);
  fourdconjugategradient->SetForwardProjectionFilter(ConjugateGradientFilterType::FP_CUDARAYCAST);
  fourdconjugategradient->SetBackProjectionFilter(ConjugateGradientFilterType::BP_CUDAVOXELBASED);
#else
  fourdconjugategradient->SetForwardProjectionFilter(ConjugateGradientFilterType::FP_JOSEPH);
  fourdconjugategradient->SetBackProjectionFilter(ConjugateGradientFilterType::BP_VOXELBASED);
#endif

  // Set the newly ordered arguments
  fourdconjugategradient->SetInputProjectionStack(reorder->GetOutput());
  fourdconjugategradient->SetGeometry(reorder->GetOutputGeometry());
  fourdconjugategradient->SetWeights(signalToInterpolationWeights->GetOutput());
  fourdconjugategradient->SetSignal(reorder->GetOutputSignal());

  auto verboseIterationCommand = rtk::VerboseIterationCommand<ConjugateGradientFilterType>::New();
  fourdconjugategradient->AddObserver(itk::AnyEvent(), verboseIterationCommand);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(fourdconjugategradient->Update())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(fourdconjugategradient->GetOutput(), "fourdconjugategradient.mha"));

  return EXIT_SUCCESS;
}
