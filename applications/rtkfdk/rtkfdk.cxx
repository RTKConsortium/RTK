/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtkfdk_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkDisplacedDetectorForOffsetFieldOfViewImageFilter.h"
#include "rtkParkerShortScanImageFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#ifdef RTK_USE_CUDA
#  include "rtkCudaDisplacedDetectorImageFilter.h"
// TODO #  include "rtkCudaDisplacedDetectorForOffsetFieldOfViewImageFilter.h"
#  include "rtkCudaParkerShortScanImageFilter.h"
#  include "rtkCudaFDKConeBeamReconstructionFilter.h"
#endif
#include "rtkFDKWarpBackProjectionImageFilter.h"
#include "rtkCyclicDeformationImageFilter.h"
#include "rtkProgressCommands.h"

#include <itkStreamingImageFilter.h>
#include <itkImageRegionSplitterDirection.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkfdk, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

  using CPUOutputImageType = itk::Image<OutputPixelType, Dimension>;
#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = CPUOutputImageType;
#endif

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkfdk>(reader, args_info);

  if (!args_info.lowmem_flag)
  {
    if (args_info.verbose_flag)
      std::cout << "Reading... " << std::endl;
    TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update())
  }

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

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
  if (!strcmp(args_info.hardware_arg, "cuda"))
    ddf = DDFType::New();
  else
    ddf = rtk::DisplacedDetectorForOffsetFieldOfViewImageFilter<OutputImageType>::New();
  ddf->SetInput(reader->GetOutput());
  ddf->SetGeometry(geometry);
  ddf->SetDisable(args_info.nodisplaced_flag);

  // Short scan image filter
  using PSSFCPUType = rtk::ParkerShortScanImageFilter<OutputImageType>;
#ifdef RTK_USE_CUDA
  using PSSFType = rtk::CudaParkerShortScanImageFilter;
#else
  using PSSFType = rtk::ParkerShortScanImageFilter<OutputImageType>;
#endif
  PSSFCPUType::Pointer pssf;
  if (!strcmp(args_info.hardware_arg, "cuda"))
    pssf = PSSFType::New();
  else
    pssf = PSSFCPUType::New();
  pssf->SetInput(ddf->GetOutput());
  pssf->SetGeometry(geometry);
  pssf->InPlaceOff();
  pssf->SetAngularGapThreshold(args_info.short_arg * itk::Math::pi / 180.);

  // Create reconstructed image
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkfdk>(constantImageSource, args_info);

  // Motion-compensated objects for the compensation of a cyclic deformation.
  // Although these will only be used if the command line options for motion
  // compensation are set, we still create the object before hand to avoid auto
  // destruction.
  using DVFPixelType = itk::Vector<float, 3>;
  using DeformationType = rtk::CyclicDeformationImageFilter<itk::Image<DVFPixelType, 4>, itk::Image<DVFPixelType, 3>>;
  auto dvfReader = itk::ImageFileReader<DeformationType::InputImageType>::New();
  auto def = DeformationType::New();
  def->SetInput(dvfReader->GetOutput());
  auto bp = rtk::FDKWarpBackProjectionImageFilter<OutputImageType, OutputImageType, DeformationType>::New();
  bp->SetDeformation(def);
  bp->SetGeometry(geometry);

  // This macro sets options for fdk filter which I can not see how to do better
  // because TFFTPrecision is not the same, e.g. for CPU and CUDA (SR)
#define SET_FELDKAMP_OPTIONS(f)                                   \
  f->SetInput(0, constantImageSource->GetOutput());               \
  f->SetInput(1, pssf->GetOutput());                              \
  f->SetGeometry(geometry);                                       \
  f->GetRampFilter()->SetTruncationCorrection(args_info.pad_arg); \
  f->GetRampFilter()->SetHannCutFrequency(args_info.hann_arg);    \
  f->GetRampFilter()->SetHannCutFrequencyY(args_info.hannY_arg);  \
  f->SetProjectionSubsetSize(args_info.subsetsize_arg);           \
  if (args_info.verbose_flag)                                     \
  {                                                               \
    f->AddObserver(itk::AnyEvent(), progressCommand);             \
  }

  // FDK reconstruction filtering
  using FDKCPUType = rtk::FDKConeBeamReconstructionFilter<OutputImageType>;
  FDKCPUType::Pointer feldkamp;
#ifdef RTK_USE_CUDA
  using FDKCUDAType = rtk::CudaFDKConeBeamReconstructionFilter;
  FDKCUDAType::Pointer feldkampCUDA;
#endif

  itk::Image<OutputPixelType, Dimension> * pfeldkamp = nullptr;
  if (!strcmp(args_info.hardware_arg, "cpu"))
  {
    // Progress reporting
    using PercentageProgressCommandType = rtk::PercentageProgressCommand<FDKCPUType>;
    auto progressCommand = PercentageProgressCommandType::New();

    feldkamp = FDKCPUType::New();
    SET_FELDKAMP_OPTIONS(feldkamp);

    // Motion compensated CBCT settings
    if (args_info.signal_given && args_info.dvf_given)
    {
      dvfReader->SetFileName(args_info.dvf_arg);
      def->SetSignalFilename(args_info.signal_arg);
      feldkamp->SetBackProjectionFilter(bp.GetPointer());
    }
    pfeldkamp = feldkamp->GetOutput();
  }
#ifdef RTK_USE_CUDA
  else if (!strcmp(args_info.hardware_arg, "cuda"))
  {
    // Motion compensation not supported in cuda
    if (args_info.signal_given && args_info.dvf_given)
    {
      std::cerr << "Motion compensation is not supported in CUDA. Aborting" << std::endl;
      return EXIT_FAILURE;
    }

    // Progress reporting
    using PercentageProgressCommandType = rtk::PercentageProgressCommand<FDKCUDAType>;
    auto progressCommand = PercentageProgressCommandType::New();

    feldkampCUDA = FDKCUDAType::New();
    SET_FELDKAMP_OPTIONS(feldkampCUDA);
    pfeldkamp = feldkampCUDA->GetOutput();
  }
#endif

  // Streaming depending on streaming capability of writer
  auto streamerBP = itk::StreamingImageFilter<CPUOutputImageType, CPUOutputImageType>::New();
  streamerBP->SetInput(pfeldkamp);
  streamerBP->SetNumberOfStreamDivisions(args_info.divisions_arg);
  auto splitter = itk::ImageRegionSplitterDirection::New();
  splitter->SetDirection(2); // Prevent splitting along z axis. As a result, splitting will be performed along y axis
  streamerBP->SetRegionSplitter(splitter);

  // Write
  if (args_info.verbose_flag)
    std::cout << "Reconstructing and writing... " << std::endl;

  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(streamerBP->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
