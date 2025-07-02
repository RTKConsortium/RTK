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

#include "rtkfourdfdk_ggo.h"
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
#include "rtkSelectOneProjectionPerCycleImageFilter.h"
#include <itkImageRegionConstIterator.h>
#include <itkStreamingImageFilter.h>
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkfourdfdk, args_info);

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
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkfourdfdk>(reader, args_info);

  // Geometry
  if (args_info.verbose_flag)
    std::cout << "Reading geometry information from " << args_info.geometry_arg << "..." << std::endl;
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometry = rtk::ReadGeometry(args_info.geometry_arg));

  // Part specific to 4D
  auto selector = rtk::SelectOneProjectionPerCycleImageFilter<OutputImageType>::New();
  selector->SetInput(reader->GetOutput());
  selector->SetInputGeometry(geometry);
  selector->SetSignalFilename(args_info.signal_arg);

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
  if (!strcmp(args_info.hardware_arg, "cuda"))
    pssf = PSSFType::New();
  else
    pssf = PSSFCPUType::New();
  pssf->SetInput(ddf->GetOutput());
  pssf->SetGeometry(selector->GetOutputGeometry());
  pssf->InPlaceOff();

  // Create one frame of the reconstructed image
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkfourdfdk>(constantImageSource, args_info);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(constantImageSource->Update())

  // This macro sets options for fdk filter which I can not see how to do better
  // because TFFTPrecision is not the same, e.g. for CPU and CUDA (SR)
#define SET_FELDKAMP_OPTIONS(f)                                   \
  f->SetInput(0, constantImageSource->GetOutput());               \
  f->SetInput(1, pssf->GetOutput());                              \
  f->SetGeometry(selector->GetOutputGeometry());                  \
  f->GetRampFilter()->SetTruncationCorrection(args_info.pad_arg); \
  f->GetRampFilter()->SetHannCutFrequency(args_info.hann_arg);    \
  f->GetRampFilter()->SetHannCutFrequencyY(args_info.hannY_arg);  \
  f->SetProjectionSubsetSize(args_info.subsetsize_arg)

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
    feldkamp = FDKCPUType::New();
    SET_FELDKAMP_OPTIONS(feldkamp);

    pfeldkamp = feldkamp->GetOutput();
  }
#ifdef RTK_USE_CUDA
  else if (!strcmp(args_info.hardware_arg, "cuda"))
  {
    feldkampCUDA = FDKCUDAType::New();
    SET_FELDKAMP_OPTIONS(feldkampCUDA);
    pfeldkamp = feldkampCUDA->GetOutput();
  }
#endif

  // Streaming depending on streaming capability of writer
  auto streamerBP = itk::StreamingImageFilter<CPUOutputImageType, CPUOutputImageType>::New();
  streamerBP->SetInput(pfeldkamp);
  streamerBP->SetNumberOfStreamDivisions(args_info.divisions_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(streamerBP->UpdateOutputInformation())

  // Create empty 4D image
  using FourDOutputImageType = itk::Image<OutputPixelType, Dimension + 1>;
  using FourDConstantImageSourceType = rtk::ConstantImageSource<FourDOutputImageType>;
  auto fourDConstantImageSource = FourDConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<FourDConstantImageSourceType, args_info_rtkfourdfdk>(fourDConstantImageSource,
                                                                                          args_info);

  // GenGetOpt can't handle default arguments for multiple arguments like size or spacing.
  // The only default it accepts is to set all components of a multiple argument to the same value.
  // Default size is 256^4, ie the number of reconstructed instants is 256. It has to be set to a more reasonable
  // value which is why a "frames" argument is introduced
  FourDConstantImageSourceType::SizeType fourDInputSize(fourDConstantImageSource->GetSize());
  fourDInputSize[3] = args_info.frames_arg;
  fourDConstantImageSource->SetSize(fourDInputSize);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(fourDConstantImageSource->Update())

  // Go over each frame, reconstruct 3D frame and paste with iterators in 4D image
  for (int f = 0; f < args_info.frames_arg; f++)
  {
    if (args_info.verbose_flag)
      std::cout << "Reconstructing frame #" << f << "..." << std::endl;
    selector->SetPhase(f / (double)args_info.frames_arg);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(streamerBP->UpdateLargestPossibleRegion())

    FourDConstantImageSourceType::OutputImageRegionType region;
    region = fourDConstantImageSource->GetOutput()->GetLargestPossibleRegion();
    region.SetIndex(3, f);
    region.SetSize(3, 1);

    itk::ImageRegionIterator<FourDOutputImageType> it4D(fourDConstantImageSource->GetOutput(), region);
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
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(fourDConstantImageSource->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
