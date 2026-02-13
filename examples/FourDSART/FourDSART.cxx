#include "rtkFourDSARTConeBeamReconstructionFilter.h"
#include "rtkIterationCommands.h"
#include "rtkSignalToInterpolationWeights.h"
#include "rtkReorderProjectionsImageFilter.h"
#include "rtkProjectionsReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFileReader.h"

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

  // Generate the input volume series, used as initial estimate by 4D conjugate gradient
  auto fourDOrigin = itk::MakePoint(-63., -31., -63., 0.);
  auto fourDSpacing = itk::MakeVector(4., 4., 4., 1.);
  auto fourDSize = itk::MakeSize(32, 16, 32, 8);

  using ConstantFourDSourceType = rtk::ConstantImageSource<VolumeSeriesType>;
  auto fourDSource = ConstantFourDSourceType::New();

  fourDSource->SetOrigin(fourDOrigin);
  fourDSource->SetSpacing(fourDSpacing);
  fourDSource->SetSize(fourDSize);
  fourDSource->SetConstant(0.);
  fourDSource->Update();

  // Read geometry, projections and signal
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry("four_d_geometry.xml"));

  using ReaderType = rtk::ProjectionsReader<ProjectionStackType>;
  auto                     projectionsReader = ReaderType::New();
  std::vector<std::string> fileNames = std::vector<std::string>();
  fileNames.push_back("four_d_projections.mha");
  projectionsReader->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(projectionsReader->Update());

  // Re-order geometry and projections
  // In the new order, projections with identical phases are packed together
  std::vector<double> signal = rtk::ReadSignalFile("four_d_signal.txt");
  auto                reorder = rtk::ReorderProjectionsImageFilter<ProjectionStackType>::New();
  reorder->SetInput(projectionsReader->GetOutput());
  reorder->SetInputGeometry(geometry);
  reorder->SetInputSignal(signal);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reorder->Update())

  // Release the memory holding the stack of original projections
  projectionsReader->GetOutput()->ReleaseData();

  // Compute the interpolation weights
  auto signalToInterpolationWeights = rtk::SignalToInterpolationWeights::New();
  signalToInterpolationWeights->SetSignal(reorder->GetOutputSignal());
  signalToInterpolationWeights->SetNumberOfReconstructedFrames(fourDSize[3]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(signalToInterpolationWeights->Update())

  // Set the forward and back projection filters to be used
  using ConjugateGradientFilterType = rtk::FourDSARTConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>;
  auto FourDSART = ConjugateGradientFilterType::New();

  FourDSART->SetInputVolumeSeries(fourDSource->GetOutput());
  FourDSART->SetNumberOfIterations(2);
  FourDSART->SetNumberOfProjectionsPerSubset(16);
  FourDSART->SetLambda(0.3);

#ifdef RTK_USE_CUDA
  FourDSART->SetForwardProjectionFilter(ConjugateGradientFilterType::FP_CUDARAYCAST);
  FourDSART->SetBackProjectionFilter(ConjugateGradientFilterType::BP_CUDAVOXELBASED);
#else
  FourDSART->SetForwardProjectionFilter(ConjugateGradientFilterType::FP_JOSEPH);
  FourDSART->SetBackProjectionFilter(ConjugateGradientFilterType::BP_VOXELBASED);
#endif

  // Set the newly ordered arguments
  FourDSART->SetInputProjectionStack(reorder->GetOutput());
  FourDSART->SetGeometry(reorder->GetOutputGeometry());
  FourDSART->SetWeights(signalToInterpolationWeights->GetOutput());
  FourDSART->SetSignal(reorder->GetOutputSignal());

  auto verboseIterationCommand = rtk::VerboseIterationCommand<ConjugateGradientFilterType>::New();
  FourDSART->AddObserver(itk::AnyEvent(), verboseIterationCommand);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(FourDSART->Update())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(FourDSART->GetOutput(), "fourdsart.mha"));

  return EXIT_SUCCESS;
}
