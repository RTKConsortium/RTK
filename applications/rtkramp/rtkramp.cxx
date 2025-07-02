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

#include "rtkramp_ggo.h"
#include "rtkMacro.h"
#include "rtkGgoFunctions.h"
#include "rtkFFTRampImageFilter.h"
#include "rtkConfiguration.h"
#ifdef RTK_USE_CUDA
#  include "rtkCudaFFTRampImageFilter.h"
#endif

#include <itkImageFileWriter.h>
#include <itkStreamingImageFilter.h>
#include "rtkProgressCommands.h"

int
main(int argc, char * argv[])
{
  GGO(rtkramp, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

  // Projections reader
  using ReaderType = rtk::ProjectionsReader<OutputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkramp>(reader, args_info);
  if (!args_info.lowmem_flag)
  {
    if (args_info.verbose_flag)
      std::cout << "Reading... " << std::endl;
    TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update())
  }
  else
    TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->UpdateOutputInformation())

  // Ramp filter
#ifdef RTK_USE_CUDA
  using CudaRampFilterType = rtk::CudaFFTRampImageFilter;
  CudaRampFilterType::Pointer cudaRampFilter;
#endif
  using CPURampFilterType = rtk::FFTRampImageFilter<OutputImageType, OutputImageType, double>;
  CPURampFilterType::Pointer rampFilter;
  if (!strcmp(args_info.hardware_arg, "cuda"))
  {
#ifdef RTK_USE_CUDA
    cudaRampFilter = CudaRampFilterType::New();
    cudaRampFilter->SetInput(reader->GetOutput());
    cudaRampFilter->SetTruncationCorrection(args_info.pad_arg);
    cudaRampFilter->SetHannCutFrequency(args_info.hann_arg);
    cudaRampFilter->SetHannCutFrequencyY(args_info.hannY_arg);
#else
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
#endif
  }
  else
  {
    rampFilter = CPURampFilterType::New();
    rampFilter->SetInput(reader->GetOutput());
    rampFilter->SetTruncationCorrection(args_info.pad_arg);
    rampFilter->SetHannCutFrequency(args_info.hann_arg);
    rampFilter->SetHannCutFrequencyY(args_info.hannY_arg);
  }

  // Streaming filter
  auto streamer = itk::StreamingImageFilter<OutputImageType, OutputImageType>::New();
#ifdef RTK_USE_CUDA
  if (!strcmp(args_info.hardware_arg, "cuda"))
    streamer->SetInput(cudaRampFilter->GetOutput());
  else
#endif
    streamer->SetInput(rampFilter->GetOutput());
  streamer->SetNumberOfStreamDivisions(1 + reader->GetOutput()->GetLargestPossibleRegion().GetSize(2) /
                                             args_info.subsetsize_arg);

  if (args_info.verbose_flag)
  {
    auto progressCommand = rtk::PercentageProgressCommand<CPURampFilterType>::New();
#ifdef RTK_USE_CUDA
    if (!strcmp(args_info.hardware_arg, "cuda"))
      cudaRampFilter->AddObserver(itk::ProgressEvent(), progressCommand);
    else
#endif
      rampFilter->AddObserver(itk::ProgressEvent(), progressCommand);
  }

  TRY_AND_EXIT_ON_ITK_EXCEPTION(streamer->Update())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(streamer->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
