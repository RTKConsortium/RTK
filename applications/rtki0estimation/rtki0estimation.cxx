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

#include "rtki0estimation_ggo.h"
#include "rtkMacro.h"
#include "rtkGgoFunctions.h"

#include <itkExtractImageFilter.h>
#include "rtkI0EstimationProjectionFilter.h"
#include "rtkProjectionsReader.h"

#include <vector>
#include <algorithm>
#include <string>

int main(int argc, char *argv[])
{
  GGO(rtki0estimation, args_info);

  typedef unsigned short InputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< InputPixelType, Dimension > InputImageType;
  typedef itk::Image< unsigned, Dimension >       OutputHistogramType;

  typedef rtk::ProjectionsReader< InputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( rtk::GetProjectionsFileNamesFromGgo(args_info) );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->UpdateOutputInformation() )

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

  typedef rtk::I0EstimationProjectionFilter< InputImageType, InputImageType, 2 > I0FilterType;

  std::vector< unsigned short > I0buffer;

  int istep         = 1;
  unsigned int imin = 1;
  unsigned int imax = (unsigned int)subsetRegion.GetSize()[2];
  if ( args_info.range_given )
    {
    if ( ( args_info.range_arg[0] <= args_info.range_arg[2] )
         && ( istep <= ( args_info.range_arg[2] - args_info.range_arg[0] ) ) )
      {
      imin = args_info.range_arg[0];
      istep = args_info.range_arg[1];
      imax = std::min(unsigned(args_info.range_arg[2]), imax);
      }
    }

  I0FilterType::Pointer i0est = I0FilterType::New();

  if ( args_info.lambda_given )
    {
    i0est->SetLambda(args_info.lambda_arg);
    }
  if ( args_info.expected_arg != 65535 )
    {
    i0est->SetExpectedI0(args_info.expected_arg);
    }
  i0est->SaveHistogramsOn();

  for ( unsigned int i = imin; i < imax; i += istep )
    {
    i0est->SetInput( extract->GetOutput() );

    start[2] = i;
    InputImageType::RegionType desiredRegion(start, extractSize);
    extract->SetExtractionRegion(desiredRegion);

    try
      {
      TRY_AND_EXIT_ON_ITK_EXCEPTION( i0est->UpdateLargestPossibleRegion() )
      }
    catch ( itk::ExceptionObject & err )
      {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
      }

    I0buffer.push_back( i0est->GetI0() );
    I0buffer.push_back( i0est->GetI0rls() );
    I0buffer.push_back( i0est->GetI0fwhm() );
    }

  if ( args_info.debug_given )
    {
    std::ofstream paramFile;
    paramFile.open(args_info.debug_arg);
    std::vector< unsigned short >::const_iterator it = I0buffer.begin();
    for (; it != I0buffer.end(); ++it )
      {
      paramFile << *it << ",";
      }
    paramFile.close();
    }

  return EXIT_SUCCESS;
}
