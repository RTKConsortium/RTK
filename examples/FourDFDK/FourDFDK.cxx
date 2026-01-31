#include "rtkConfiguration.h"
#include "rtkThreeDCircularProjectionGeometryXMLFileReader.h"
#include "rtkDisplacedDetectorForOffsetFieldOfViewImageFilter.h"
#include "rtkParkerShortScanImageFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#ifdef RTK_USE_CUDA
#  include "rtkCudaDisplacedDetectorImageFilter.h"
// TODO #  include "rtkCudaDisplacedDetectorForOffsetFieldOfViewImageFilter.h"
#  include "rtkCudaParkerShortScanImageFilter.h"
#  include "rtkCudaFDKConeBeamReconstructionFilter.h"
#endif
#include "rtkSelectOneProjectionPerCycleImageFilter.h"
#include "rtkProjectionsReader.h"

#ifdef RTK_USE_CUDA
#  include <itkCudaImage.h>
#endif
#include <itkImageRegionConstIterator.h>
#include <itkStreamingImageFilter.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

  using CPUOutputImageType = itk::Image<OutputPixelType, Dimension>;
#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = CPUOutputImageType;
#endif

  // Read geometry, projections and signal
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry("four_d_geometry.xml"));

  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto                     projectionsReader = ReaderType::New();
  std::vector<std::string> fileNames = std::vector<std::string>();
  fileNames.push_back("four_d_projections.mha");
  projectionsReader->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(projectionsReader->Update());

  // Part specific to 4D
  auto selector = rtk::SelectOneProjectionPerCycleImageFilter<OutputImageType>::New();
  selector->SetInput(projectionsReader->GetOutput());
  selector->SetInputGeometry(geometry);
  selector->SetSignalFilename("four_d_signal.txt");

  // Check on hardware parameter
#ifndef RTK_USE_CUDA
  if (!strcmp(args_info.hardware_arg, "cuda"))
  {
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
  }
#endif

  // Displaced detector weighting
#ifdef RTK_USE_CUDA
  using DDFType = rtk::CudaDisplacedDetectorImageFilter;
#else
  using DDFType = rtk::DisplacedDetectorForOffsetFieldOfViewImageFilter<OutputImageType>;
#endif

  rtk::DisplacedDetectorImageFilter<OutputImageType>::Pointer ddf;
#ifdef RTK_USE_CUDA
  ddf = DDFType::New();
#else
  ddf = rtk::DisplacedDetectorForOffsetFieldOfViewImageFilter<OutputImageType>::New();
#endif
  ddf->SetInput(selector->GetOutput());
  ddf->SetGeometry(selector->GetOutputGeometry());

  // Short scan image filter
  using PSSFCPUType = rtk::ParkerShortScanImageFilter<OutputImageType>;
#ifdef RTK_USE_CUDA
  using PSSFType = rtk::CudaParkerShortScanImageFilter;
#else
  using PSSFType = rtk::ParkerShortScanImageFilter<OutputImageType>;
#endif
  PSSFCPUType::Pointer pssf;

#ifdef RTK_USE_CUDA
  pssf = PSSFType::New();
#else
  pssf = PSSFCPUType::New();
#endif
  pssf->SetInput(ddf->GetOutput());
  pssf->SetGeometry(selector->GetOutputGeometry());
  pssf->InPlaceOff();

  // Create one frame of the reconstructed image
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto constantImageSource = ConstantImageSourceType::New();
  constantImageSource->SetOrigin(itk::MakePoint(-63., -31., -63.));
  constantImageSource->SetSpacing(itk::MakeVector(4., 4., 4.));
  constantImageSource->SetSize(itk::MakeSize(32, 16, 32));
  constantImageSource->SetConstant(0.);
  constantImageSource->Update();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantImageSource->Update())

  // FDK reconstruction filtering
#ifdef RTK_USE_CUDA
  using FDKType = rtk::CudaFDKConeBeamReconstructionFilter;
#else
  using FDKType = rtk::FDKConeBeamReconstructionFilter<OutputImageType>;
#endif
  auto feldkamp = FDKType::New();
  feldkamp->SetInput(0, constantImageSource->GetOutput());
  feldkamp->SetInput(1, pssf->GetOutput());
  feldkamp->SetGeometry(selector->GetOutputGeometry());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(feldkamp->Update());

  // Streaming depending on streaming capability of writer
  auto streamerBP = itk::StreamingImageFilter<CPUOutputImageType, CPUOutputImageType>::New();
  streamerBP->SetInput(feldkamp->GetOutput());
  streamerBP->SetNumberOfStreamDivisions(2);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(streamerBP->UpdateOutputInformation())

  // Create empty 4D image
  using FourDOutputImageType = itk::Image<OutputPixelType, Dimension + 1>;
  using ConstantFourDSourceType = rtk::ConstantImageSource<FourDOutputImageType>;
  auto fourDSource = ConstantFourDSourceType::New();
  fourDSource->SetOrigin(itk::MakePoint(-63., -31., -63., 0.));
  fourDSource->SetSpacing(itk::MakeVector(4., 4., 4., 1.));
  fourDSource->SetSize(itk::MakeSize(32, 16, 32, 4));
  fourDSource->SetConstant(0.);
  fourDSource->Update();

  // Go over each frame, reconstruct 3D frame and paste with iterators in 4D image
  for (int f = 0; f < 4; f++)
  {
    selector->SetPhase(f / (double)4);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(streamerBP->UpdateLargestPossibleRegion())

    ConstantFourDSourceType::OutputImageRegionType region;
    region = fourDSource->GetOutput()->GetLargestPossibleRegion();
    region.SetIndex(3, f);
    region.SetSize(3, 1);

    itk::ImageRegionIterator<FourDOutputImageType> it4D(fourDSource->GetOutput(), region);
    itk::ImageRegionIterator<CPUOutputImageType>   it3D(streamerBP->GetOutput(),
                                                      streamerBP->GetOutput()->GetLargestPossibleRegion());
    while (!it3D.IsAtEnd())
    {
      it4D.Set(it3D.Get());
      ++it4D;
      ++it3D;
    }
  }

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(fourDSource->GetOutput(), "fourdfdk.mha"));

  return EXIT_SUCCESS;
}
