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

#include "rtkramp_ggo.h"
#include "rtkMacro.h"

#include "rtkProjectionsReader.h"
#include "rtkFFTRampImageFilter.h"

#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkStreamingImageFilter.h>
#include <itkTimeProbe.h>

int main(int argc, char * argv[])
{
  GGO(rtkramp, args_info);

  typedef double OutputPixelType;
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
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );

  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )

  // Ramp filter
  typedef rtk::FFTRampImageFilter<OutputImageType> rampFilterType;
  rampFilterType::Pointer rampFilter = rampFilterType::New();
  rampFilter->SetInput( reader->GetOutput() );
  rampFilter->SetTruncationCorrection(args_info.pad_arg);
  rampFilter->SetHannCutFrequency(args_info.hann_arg);

  // Streaming filter
  typedef itk::StreamingImageFilter<OutputImageType, OutputImageType> StreamerType;
  StreamerType::Pointer streamer = StreamerType::New();
  streamer->SetInput( rampFilter->GetOutput() );
  streamer->SetNumberOfStreamDivisions( 1 + reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() / (1024*1024*4) );

  itk::TimeProbe probe;
  probe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( streamer->Update() )
  probe.Stop();
  std::cout << "The streamed ramp filter update took "
            << probe.GetMean()
            << probe.GetUnit()
            << std::endl;

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( streamer->GetOutput() );
  writer->UpdateOutputInformation();

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
