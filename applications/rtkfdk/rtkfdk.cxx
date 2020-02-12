/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
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
  ReaderType::Pointer reader = ReaderType::New();
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
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(geometryReader->GenerateOutputInformation())

  // Check on hardware parameter
#ifndef RTK_USE_CUDA
  if (!strcmp(args_info.hardware_arg, "cuda"))
  {
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
  }
#endif

  // Displaced detector weighting
  using DDFCPUType = rtk::DisplacedDetectorImageFilter<OutputImageType>;
  using DDFOFFFOVType = rtk::DisplacedDetectorForOffsetFieldOfViewImageFilter<OutputImageType>;
#ifdef RTK_USE_CUDA
  using DDFType = rtk::CudaDisplacedDetectorImageFilter;
#else
  using DDFType = rtk::DisplacedDetectorForOffsetFieldOfViewImageFilter<OutputImageType>;
#endif
  DDFCPUType::Pointer ddf;
  if (!strcmp(args_info.hardware_arg, "cuda"))
    ddf = DDFType::New();
  else
    ddf = DDFOFFFOVType::New();
  ddf->SetInput(reader->GetOutput());
  ddf->SetGeometry(geometryReader->GetOutputObject());
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
  pssf->SetGeometry(geometryReader->GetOutputObject());
  pssf->InPlaceOff();
  pssf->SetAngularGapThreshold(args_info.short_arg * itk::Math::pi / 180.);

  // Create reconstructed image
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkfdk>(constantImageSource, args_info);

  // Motion-compensated objects for the compensation of a cyclic deformation.
  // Although these will only be used if the command line options for motion
  // compensation are set, we still create the object before hand to avoid auto
  // destruction.
  using DVFPixelType = itk::Vector<float, 3>;
  using DVFImageSequenceType = itk::Image<DVFPixelType, 4>;
  using DVFImageType = itk::Image<DVFPixelType, 3>;
  using DeformationType = rtk::CyclicDeformationImageFilter<DVFImageSequenceType, DVFImageType>;
  using DVFReaderType = itk::ImageFileReader<DeformationType::InputImageType>;
  DVFReaderType::Pointer   dvfReader = DVFReaderType::New();
  DeformationType::Pointer def = DeformationType::New();
  def->SetInput(dvfReader->GetOutput());
  using WarpBPType = rtk::FDKWarpBackProjectionImageFilter<OutputImageType, OutputImageType, DeformationType>;
  WarpBPType::Pointer bp = WarpBPType::New();
  bp->SetDeformation(def);
  bp->SetGeometry(geometryReader->GetOutputObject());

  // This macro sets options for fdk filter which I can not see how to do better
  // because TFFTPrecision is not the same, e.g. for CPU and CUDA (SR)
#define SET_FELDKAMP_OPTIONS(f)                                                                                        \
  f->SetInput(0, constantImageSource->GetOutput());                                                                    \
  f->SetInput(1, pssf->GetOutput());                                                                                   \
  f->SetGeometry(geometryReader->GetOutputObject());                                                                   \
  f->GetRampFilter()->SetTruncationCorrection(args_info.pad_arg);                                                      \
  f->GetRampFilter()->SetHannCutFrequency(args_info.hann_arg);                                                         \
  f->GetRampFilter()->SetHannCutFrequencyY(args_info.hannY_arg);                                                       \
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

    feldkampCUDA = FDKCUDAType::New();
    SET_FELDKAMP_OPTIONS(feldkampCUDA);
    pfeldkamp = feldkampCUDA->GetOutput();
  }
#endif

  // Write
  using WriterType = itk::ImageFileWriter<CPUOutputImageType>;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(pfeldkamp);
  writer->SetNumberOfStreamDivisions(args_info.divisions_arg);

  if (args_info.verbose_flag)
    std::cout << "Reconstructing and writing... " << std::endl;

  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  return EXIT_SUCCESS;
}
