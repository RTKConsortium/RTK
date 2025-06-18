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

#include "rtkscatterglarecorrection_ggo.h"
#include "rtkMacro.h"
#include "rtkGgoFunctions.h"

#include <itkExtractImageFilter.h>
#include <itkSubtractImageFilter.h>
#ifdef RTK_USE_CUDA
#  include "rtkCudaScatterGlareCorrectionImageFilter.h"
#else
#  include "rtkScatterGlareCorrectionImageFilter.h"
#endif

#include "rtkProjectionsReader.h"
#include <itkPasteImageFilter.h>
#include <rtkConstantImageSource.h>
#include <itkImageFileWriter.h>

#include <vector>
#include <algorithm>
#include <string>

int
main(int argc, char * argv[])
{
  GGO(rtkscatterglarecorrection, args_info);

  using InputPixelType = float;
  constexpr unsigned int Dimension = 3;
#ifdef RTK_USE_CUDA
  using InputImageType = itk::CudaImage<InputPixelType, Dimension>;
#else
  using InputImageType = itk::Image<InputPixelType, Dimension>;
#endif

  using ReaderType = rtk::ProjectionsReader<InputImageType>; // Warning: preprocess images
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames(rtk::GetProjectionsFileNamesFromGgo(args_info));
  reader->ComputeLineIntegralOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->UpdateOutputInformation())

  // Input projection parameters
  InputImageType::SizeType sizeInput = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
  int                      Nproj = sizeInput[2];


  std::vector<float> coef;
  if (args_info.coefficients_given == 2)
  {
    coef.push_back(args_info.coefficients_arg[0]);
    coef.push_back(args_info.coefficients_arg[1]);
  }
  else
  {
    std::cerr << "--coefficients requires exactly 2 coefficients" << std::endl;
    return EXIT_FAILURE;
  }

#ifdef RTK_USE_CUDA
  using ScatterCorrectionType = rtk::CudaScatterGlareCorrectionImageFilter;
#else
  using ScatterCorrectionType = rtk::ScatterGlareCorrectionImageFilter<InputImageType, InputImageType, float>;
#endif
  ScatterCorrectionType::Pointer SFilter = ScatterCorrectionType::New();
  SFilter->SetTruncationCorrection(0.0);
  SFilter->SetCoefficients(coef);

  using ConstantImageSourceType = rtk::ConstantImageSource<InputImageType>;
  ConstantImageSourceType::Pointer constantSource = ConstantImageSourceType::New();

  using PasteImageFilterType = itk::PasteImageFilter<InputImageType, InputImageType>;
  PasteImageFilterType::Pointer paste = PasteImageFilterType::New();
  paste->SetSourceImage(SFilter->GetOutput());
  paste->SetDestinationImage(constantSource->GetOutput());

  std::cout << "Starting processing" << std::endl;
  int  projid = 0;
  bool first = true;
  while (projid < Nproj)
  {
    int curBufferSize = std::min(args_info.bufferSize_arg, Nproj - projid);

    InputImageType::RegionType sliceRegionA = reader->GetOutput()->GetLargestPossibleRegion();
    InputImageType::RegionType desiredRegionA;
    desiredRegionA.SetSize(itk::MakeSize(sliceRegionA.GetSize()[0], sliceRegionA.GetSize()[1], curBufferSize));
    desiredRegionA.SetIndex(itk::MakeIndex(sliceRegionA.GetIndex()[0], sliceRegionA.GetIndex()[1], projid));

    using ExtractFilterType = itk::ExtractImageFilter<InputImageType, InputImageType>;
    ExtractFilterType::Pointer extract = ExtractFilterType::New();
    extract->SetDirectionCollapseToIdentity();
    extract->SetExtractionRegion(desiredRegionA);
    extract->SetInput(reader->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(extract->Update())

    InputImageType::Pointer image = extract->GetOutput();
    image->DisconnectPipeline();

    SFilter->SetInput(image);
    SFilter->GetOutput()->SetRequestedRegion(image->GetRequestedRegion());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(SFilter->Update())

    InputImageType::Pointer procImage = SFilter->GetOutput();
    procImage->DisconnectPipeline();

    InputImageType::Pointer outImage;
    if (args_info.difference_flag)
    {
      using SubtractImageFilterType = itk::SubtractImageFilter<InputImageType, InputImageType>;
      SubtractImageFilterType::Pointer subtractFilter = SubtractImageFilterType::New();
      subtractFilter->SetInput1(image);
      subtractFilter->SetInput2(procImage);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(subtractFilter->Update())
      outImage = subtractFilter->GetOutput();
      outImage->DisconnectPipeline();
    }
    else
    {
      outImage = procImage;
    }

    InputImageType::IndexType current_idx = outImage->GetLargestPossibleRegion().GetIndex();
    current_idx[2] = projid;

    if (first)
    {
      // Initialization of the output volume
      InputImageType::SizeType sizeInput_local = outImage->GetLargestPossibleRegion().GetSize();
      sizeInput_local[2] = Nproj;
      InputImageType::SpacingType   spacingInput = outImage->GetSpacing();
      InputImageType::PointType     originInput = outImage->GetOrigin();
      InputImageType::DirectionType imageDirection;
      imageDirection.SetIdentity();

      constantSource->SetOrigin(originInput);
      constantSource->SetSpacing(spacingInput);
      constantSource->SetDirection(imageDirection);
      constantSource->SetSize(sizeInput_local);
      constantSource->SetConstant(0.);
      first = false;
    }
    else
    {
      paste->SetDestinationImage(paste->GetOutput());
    }

    paste->SetSourceImage(outImage);
    paste->SetSourceRegion(outImage->GetLargestPossibleRegion());

    paste->SetDestinationIndex(current_idx);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(paste->Update())

    projid += curBufferSize;
  }

  if (args_info.output_given)
  {
    TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(paste->GetOutput(), args_info.output_arg))
  }

  return EXIT_SUCCESS;
}
