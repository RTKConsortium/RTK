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

#include "rtkscatterglarecorrection_ggo.h"

#include "rtkMacro.h"
#include "rtkGgoFunctions.h"

#include <itkExtractImageFilter.h>
#include "rtkScatterGlareCorrectionImageFilter.h"
#include "rtkProjectionsReader.h"
#include <itkPasteImageFilter.h>
#include <rtkConstantImageSource.h>
#include <itkImageFileWriter.h>

#include <itkMultiplyImageFilter.h>

#include <vector>
#include <algorithm>
#include <string>

int main(int argc, char *argv[])
{
  GGO(rtkscatterglarecorrection, args_info);

  typedef float InputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< InputPixelType, Dimension > InputImageType;
  
  typedef itk::RegularExpressionSeriesFileNames RegexpType;
  RegexpType::Pointer names = RegexpType::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(args_info.nsort_flag);
  names->SetRegularExpression(args_info.regexp_arg);

  typedef rtk::ProjectionsReader< InputImageType > ReaderType;  // Warning: preprocess images
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );
  reader->UpdateOutputInformation();

  typedef itk::ExtractImageFilter< InputImageType, InputImageType > ExtractFilterType;
  ExtractFilterType::Pointer extract = ExtractFilterType::New();
  extract->InPlaceOff();
  extract->SetDirectionCollapseToSubmatrix();
  extract->SetInput( reader->GetOutput() );
  
  ExtractFilterType::InputImageRegionType subsetRegion = reader->GetOutput()->GetLargestPossibleRegion();
  InputImageType::SizeType extractSize = subsetRegion.GetSize();
  extractSize[2] = 1;
  InputImageType::IndexType start = subsetRegion.GetIndex();
  extract->SetExtractionRegion(subsetRegion);
  
  std::vector<float> coef;
  if (args_info.coefficients_given == 2) {
    coef.push_back(args_info.coefficients_arg[0]);
    coef.push_back(args_info.coefficients_arg[1]);
  }
  else {
    coef.push_back(0.0787f);
    coef.push_back(106.244f);
  }
   
  typedef rtk::ScatterGlareCorrectionImageFilter<InputImageType, InputImageType, float>   ScatterCorrectionType;
  ScatterCorrectionType::Pointer SFilter = ScatterCorrectionType::New();
  SFilter->SetInput(extract->GetOutput());
  SFilter->SetTruncationCorrection(1.0);
  SFilter->SetCoefficients(coef);
  SFilter->UpdateOutputInformation();

  typedef rtk::ConstantImageSource<InputImageType> ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantSource = ConstantImageSourceType::New();
  constantSource->SetInformationFromImage(SFilter->GetOutput());
  constantSource->SetSize(SFilter->GetOutput()->GetLargestPossibleRegion().GetSize());
  constantSource->UpdateOutputInformation();

  ConstantImageSourceType::Pointer constantSourceIn = ConstantImageSourceType::New();
  constantSourceIn->SetInformationFromImage(SFilter->GetOutput());
  constantSourceIn->SetSize(SFilter->GetOutput()->GetLargestPossibleRegion().GetSize());
  constantSourceIn->UpdateOutputInformation();
      
  typedef itk::PasteImageFilter <InputImageType, InputImageType > PasteImageFilterType;
  PasteImageFilterType::Pointer paste = PasteImageFilterType::New();
  paste->SetSourceImage(SFilter->GetOutput());
  paste->SetDestinationImage(constantSource->GetOutput());

  PasteImageFilterType::Pointer pasteIn = PasteImageFilterType::New();
  pasteIn->SetSourceImage(extract->GetOutput());
  pasteIn->SetDestinationImage(constantSourceIn->GetOutput());
      
  int istep         = 1;
  unsigned int imin = 1;
  unsigned int imax = (unsigned int)subsetRegion.GetSize()[2];
  if (args_info.range_given)
  {
    if ((args_info.range_arg[0] <= args_info.range_arg[2])
      && (istep <= (args_info.range_arg[2] - args_info.range_arg[0])))
    {
      imin = args_info.range_arg[0];
      istep = args_info.range_arg[1];
      imax = std::min(unsigned(args_info.range_arg[2]), imax);
    }
  }
  
  InputImageType::Pointer pimg;
  InputImageType::Pointer pimgIn;
  int frameoutidx = 0;
  for (unsigned int frame = (imin-1); frame < imax; frame += istep, ++frameoutidx)
  {  
    if (frame > 0) // After the first frame, use the output of paste as input
    {
      pimg = paste->GetOutput();
      pimg->DisconnectPipeline();
      paste->SetDestinationImage(pimg);

      pimgIn = pasteIn->GetOutput();
      pimgIn->DisconnectPipeline();
      pasteIn->SetDestinationImage(pimgIn);
    }
   
    start[2] = frame;
    InputImageType::RegionType desiredRegion(start, extractSize);
    extract->SetExtractionRegion(desiredRegion);

    SFilter->SetInput(extract->GetOutput());
    SFilter->UpdateLargestPossibleRegion();
        
    // Set extraction regions and indices
    InputImageType::RegionType pasteRegion = SFilter->GetOutput()->GetLargestPossibleRegion();
    pasteRegion.SetSize(Dimension - 1, 1);
    pasteRegion.SetIndex(Dimension - 1, frame);
    
    paste->SetDestinationIndex(pasteRegion.GetIndex());
    paste->SetSourceRegion(SFilter->GetOutput()->GetLargestPossibleRegion());

    paste->SetSourceImage(SFilter->GetOutput());
    paste->UpdateLargestPossibleRegion(); 

    pasteIn->SetDestinationIndex(pasteRegion.GetIndex());
    pasteIn->SetSourceRegion(extract->GetOutput()->GetLargestPossibleRegion());

    pasteIn->SetSourceImage(extract->GetOutput());
    pasteIn->UpdateLargestPossibleRegion();
  }

  typedef itk::ImageFileWriter<InputImageType> FileWriterType;
  FileWriterType::Pointer writer = FileWriterType::New();
  if (args_info.output_given) {
    writer->SetFileName(args_info.output_arg);
    writer->SetInput(paste->GetOutput());
    writer->Update();

    writer->SetFileName("input.mhd");
    writer->SetInput(pasteIn->GetOutput());
    writer->Update();
  }

  return EXIT_SUCCESS;
}
