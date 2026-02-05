#include "rtkFourDROOSTERConeBeamReconstructionFilter.h"
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
  using DVFVectorType = itk::CovariantVector<OutputPixelType, 3>;
#ifdef RTK_USE_CUDA
  using VolumeSeriesType = itk::CudaImage<OutputPixelType, 4>;
  using ProjectionStackType = itk::CudaImage<OutputPixelType, 3>;
  using VolumeType = itk::CudaImage<OutputPixelType, 3>;
  using DVFSequenceImageType = itk::CudaImage<DVFVectorType, VolumeSeriesType::ImageDimension>;
#else
  using VolumeSeriesType = itk::Image<OutputPixelType, 4>;
  using ProjectionStackType = itk::Image<OutputPixelType, 3>;
  using VolumeType = itk::Image<OutputPixelType, 3>;
  using DVFSequenceImageType = itk::Image<DVFVectorType, VolumeSeriesType::ImageDimension>;
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
  using ROOSTERFilterType = rtk::FourDROOSTERConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>;
  auto rooster = ROOSTERFilterType::New();

  rooster->SetInputVolumeSeries(fourDSource->GetOutput());
  rooster->SetCG_iterations(2);
  rooster->SetMainLoop_iterations(2);
#ifdef RTK_USE_CUDA
  rooster->SetCudaConjugateGradient(true);
  rooster->SetUseCudaCyclicDeformation(true);
  rooster->SetForwardProjectionFilter(ROOSTERFilterType::FP_CUDARAYCAST);
  rooster->SetBackProjectionFilter(ROOSTERFilterType::BP_CUDAVOXELBASED);
#else
  rooster->SetForwardProjectionFilter(ROOSTERFilterType::FP_JOSEPH);
  rooster->SetBackProjectionFilter(ROOSTERFilterType::BP_VOXELBASED);
#endif

  // Set the newly ordered arguments
  rooster->SetInputProjectionStack(reorder->GetOutput());
  rooster->SetGeometry(reorder->GetOutputGeometry());
  rooster->SetWeights(signalToInterpolationWeights->GetOutput());
  rooster->SetSignal(reorder->GetOutputSignal());

  // For each optional regularization step, set whether or not
  // it should be performed, and provide the necessary inputs

  // Positivity
  rooster->SetPerformPositivity(true);

  // Motion mask
  rooster->SetPerformMotionMask(false);

  // Spatial TV
  rooster->SetGammaTVSpace(0.1);
  rooster->SetTV_iterations(4);
  rooster->SetPerformTVSpatialDenoising(true);

  // Spatial wavelets
  rooster->SetPerformWaveletsSpatialDenoising(false);

  // Temporal TV
  rooster->SetGammaTVTime(0.1);
  rooster->SetTV_iterations(4);
  rooster->SetPerformTVTemporalDenoising(true);

  // Temporal L0
  rooster->SetPerformL0TemporalDenoising(false);

  // Total nuclear variation
  rooster->SetPerformTNVDenoising(false);

  // Warping
  rooster->SetPerformWarping(true);
  rooster->SetUseNearestNeighborInterpolationInWarping(false);
  DVFSequenceImageType::Pointer dvf;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dvf = itk::ReadImage<DVFSequenceImageType>("four_d_dvf.mha"))
  rooster->SetDisplacementField(dvf);
  rooster->SetComputeInverseWarpingByConjugateGradient(false);
  DVFSequenceImageType::Pointer idvf;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(idvf = itk::ReadImage<DVFSequenceImageType>("four_d_idvf.mha"))
  rooster->SetInverseDisplacementField(idvf);

  auto verboseIterationCommand = rtk::VerboseIterationCommand<ROOSTERFilterType>::New();
  rooster->AddObserver(itk::AnyEvent(), verboseIterationCommand);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(rooster->Update())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(rooster->GetOutput(), "fourdrooster.mha"));

  return EXIT_SUCCESS;
}
