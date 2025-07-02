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

#include "rtkgaincorrection_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"
#include "rtkConstantImageSource.h"
#include "rtkPolynomialGainCorrectionImageFilter.h"
#ifdef RTK_USE_CUDA
#  include "rtkCudaPolynomialGainCorrectionImageFilter.h"
#endif

#include <string>

#include <itkImageFileWriter.h>
#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>

int
main(int argc, char * argv[])
{
  GGO(rtkgaincorrection, args_info);

  constexpr unsigned int Dimension = 3;
#ifdef RTK_USE_CUDA
  using InputImageType = itk::CudaImage<unsigned short, Dimension>;
  using OutputImageType = itk::CudaImage<float, Dimension>;
#else
  using InputImageType = itk::Image<unsigned short, Dimension>;
  using OutputImageType = itk::Image<float, Dimension>;
#endif

  using ReaderType = rtk::ProjectionsReader<InputImageType>;
  auto reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtkgaincorrection>(reader, args_info);
  reader->ComputeLineIntegralOff(); // Don't want to preprocess data
  reader->SetFileNames(rtk::GetProjectionsFileNamesFromGgo(args_info));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->UpdateOutputInformation())

  // Input projection parameters
  int Nprojections = reader->GetOutput()->GetLargestPossibleRegion().GetSize(2);
  if (!Nprojections)
  {
    std::cout << "No DR found to process!" << std::endl;
    return EXIT_FAILURE;
  }

  // Load dark image
  std::string darkFile(args_info.calibDir_arg);
  darkFile.append("\\");
  darkFile.append(args_info.Dark_arg);

  InputImageType::Pointer darkImage;
  auto                    readerDark = itk::ImageFileReader<InputImageType>::New();
  readerDark->SetFileName(darkFile);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerDark->Update())
  darkImage = readerDark->GetOutput();
  darkImage->DisconnectPipeline();

  // Get gain image
  std::string gainFile(args_info.calibDir_arg);
  gainFile.append("\\");
  gainFile.append(args_info.Gain_arg);

  OutputImageType::Pointer gainImage;
  auto                     readerGain = itk::ImageFileReader<OutputImageType>::New();
  readerGain->SetFileName(gainFile);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerGain->Update())
  gainImage = readerGain->GetOutput();
  gainImage->DisconnectPipeline();

#ifdef RTK_USE_CUDA
  using GainType = rtk::CudaPolynomialGainCorrectionImageFilter;
#else
  using GainType = rtk::PolynomialGainCorrectionImageFilter<InputImageType, OutputImageType>;
#endif

  auto gainfilter = GainType::New();
  gainfilter->SetDarkImage(darkImage);
  gainfilter->SetGainCoefficients(gainImage);
  gainfilter->SetK(args_info.K_arg);

  // Create empty volume for storing processed images
  auto constantSource = rtk::ConstantImageSource<OutputImageType>::New();


  auto pasteFilter = itk::PasteImageFilter<OutputImageType, OutputImageType>::New();
  pasteFilter->SetDestinationImage(constantSource->GetOutput());

  int bufferSize = args_info.bufferSize_arg;
  int Nbuffers = static_cast<int>(std::ceil(static_cast<float>(Nprojections) / static_cast<float>(bufferSize)));

  bool first = true;
  for (int bid = 0; bid < Nbuffers; ++bid)
  {
    int bufferIdx = bid * bufferSize;
    int currentBufferSize = std::min(Nprojections - bufferIdx, bufferSize);

    std::cout << "Processing buffer no " << bid << " starting at image " << bufferIdx << " of size "
              << currentBufferSize << std::endl;

    InputImageType::RegionType sliceRegion = reader->GetOutput()->GetLargestPossibleRegion();
    InputImageType::RegionType desiredRegion;
    desiredRegion.SetSize(itk::MakeSize(sliceRegion.GetSize()[0], sliceRegion.GetSize()[1], currentBufferSize));
    desiredRegion.SetIndex(itk::MakeIndex(sliceRegion.GetIndex()[0], sliceRegion.GetIndex()[1], bufferIdx));

    auto extract = itk::ExtractImageFilter<InputImageType, InputImageType>::New();
    extract->SetDirectionCollapseToIdentity();
    extract->SetExtractionRegion(desiredRegion);
    extract->SetInput(reader->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(extract->Update())

    InputImageType::Pointer buffer = extract->GetOutput();
    buffer->DisconnectPipeline();

    gainfilter->SetInput(extract->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(gainfilter->Update())

    if (first)
    {
      // Initialization of the output volume
      OutputImageType::SizeType sizeInput = gainfilter->GetOutput()->GetLargestPossibleRegion().GetSize();
      sizeInput[2] = Nprojections;
      OutputImageType::SpacingType   spacingInput = gainfilter->GetOutput()->GetSpacing();
      OutputImageType::PointType     originInput = gainfilter->GetOutput()->GetOrigin();
      OutputImageType::DirectionType imageDirection;
      imageDirection.SetIdentity();

      constantSource->SetOrigin(originInput);
      constantSource->SetSpacing(spacingInput);
      constantSource->SetDirection(imageDirection);
      constantSource->SetSize(sizeInput);
      constantSource->SetConstant(0.);
    }

    OutputImageType::Pointer procBuffer = gainfilter->GetOutput();
    procBuffer->DisconnectPipeline();

    OutputImageType::IndexType current_idx = procBuffer->GetLargestPossibleRegion().GetIndex();
    current_idx[2] = bufferIdx;

    if (first)
    {
      first = false;
    }
    else
    {
      pasteFilter->SetDestinationImage(pasteFilter->GetOutput());
    }
    pasteFilter->SetSourceImage(procBuffer);
    pasteFilter->SetSourceRegion(procBuffer->GetLargestPossibleRegion());
    pasteFilter->SetDestinationIndex(current_idx);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(pasteFilter->Update())
  }

  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(pasteFilter->GetOutput(), args_info.output_arg))

  return EXIT_SUCCESS;
}
