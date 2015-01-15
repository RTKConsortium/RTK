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

#include <vector>
#include <algorithm>
#include <string>

int main(int argc, char *argv[])
{
  GGO(rtkscatterglarecorrection, args_info);

  typedef unsigned short InputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< InputPixelType, Dimension > InputImageType;
  typedef itk::Image< unsigned, Dimension >       OutputHistogramType;

  typedef itk::RegularExpressionSeriesFileNames RegexpType;
  RegexpType::Pointer names = RegexpType::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(args_info.nsort_flag);
  names->SetRegularExpression(args_info.regexp_arg);

  typedef rtk::ProjectionsReader< InputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );
  reader->UpdateOutputInformation();

  typedef itk::ExtractImageFilter< InputImageType, InputImageType > ExtractFilterType;
  ExtractFilterType::Pointer extract = ExtractFilterType::New();
  extract->InPlaceOff();
  extract->SetDirectionCollapseToSubmatrix();
  extract->SetInput( reader->GetOutput() );

  ExtractFilterType::InputImageRegionType subsetRegion = reader->GetOutput()->GetLargestPossibleRegion();
  subsetRegion = reader->GetOutput()->GetLargestPossibleRegion();
  extract->SetInput( reader->GetOutput() );
  InputImageType::SizeType extractSize = subsetRegion.GetSize();
  extractSize[2] = 1;
  InputImageType::IndexType start = subsetRegion.GetIndex();

  typedef itk::PasteImageFilter <InputImageType, InputImageType > PasteImageFilterType;
  PasteImageFilterType::Pointer pasteFilter = PasteImageFilterType::New();

  std::vector<float> coef = { 0.0787f, 106.244f };

  typedef rtk::ScatterGlareCorrectionImageFilter<InputImageType, InputImageType, float>   ScatterCorrectionType;
  ScatterCorrectionType::Pointer SFilter = ScatterCorrectionType::New();
  
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
  
  SFilter->SetTruncationCorrection(0.5);
  SFilter->SetCoefficients(coef);

  for (unsigned int i = imin; i < imax; i += istep)
  {
    std::cout << "Image no " << i << std::endl;
    SFilter->SetInput(extract->GetOutput());

    start[2] = i;
    InputImageType::RegionType desiredRegion(start, extractSize);
    extract->SetExtractionRegion(desiredRegion);

    try
    {
      SFilter->UpdateLargestPossibleRegion();
    }
    catch (itk::ExceptionObject & err)
    {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }

  }

  return EXIT_SUCCESS;
}
