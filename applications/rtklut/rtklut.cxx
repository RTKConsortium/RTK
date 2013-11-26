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

#include "rtklut_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkProjectionsReader.h"
#include "rtkLookupTableImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>

int main(int argc, char * argv[])
{
  GGO(rtklut, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Generate file names
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(false);
  names->SetRegularExpression(args_info.regexp_arg);
  names->SetSubMatch(0);

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer projections = ReaderType::New();
  projections->SetFileNames( names->GetFileNames() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( projections->Update() );

  // Read lookup table
  typedef itk::Image<OutputPixelType, 1> LUTType;
  typedef itk::ImageFileReader<LUTType> LUTReaderType;
  LUTReaderType::Pointer lutReader = LUTReaderType::New();
  lutReader->SetFileName(args_info.lut_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( lutReader->Update() );

  // Apply lookup table
  typedef rtk::LookupTableImageFilter<OutputImageType, OutputImageType> LUTFilterType;
  LUTFilterType::Pointer lutFilter = LUTFilterType::New();
  lutFilter->SetInput(projections->GetOutput());
  lutFilter->SetLookupTable(lutReader->GetOutput());
  lutFilter->Update();

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( projections->GetOutput() );
  if(args_info.verbose_flag)
    std::cout << "Writing result... " << std::endl;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
