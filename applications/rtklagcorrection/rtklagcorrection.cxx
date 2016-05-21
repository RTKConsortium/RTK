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

#include "rtklagcorrection_ggo.h"

#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#ifdef RTK_USE_CUDA
	#include "rtkCudaLagCorrectionImageFilter.h"
	#include <itkCudaImage.h>
#else 
	#include "rtkLagCorrectionImageFilter.h"
#endif

using namespace rtk;

#include <itkImageFileWriter.h>

const unsigned ModelOrder = 4;

int main(int argc, char * argv[])
{
  GGO(rtklagcorrection, args_info);

  const unsigned int Dimension = 3;
#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< unsigned short, Dimension >          OutputImageType;
#else
  typedef itk::Image< unsigned short, Dimension >              OutputImageType;
#endif
  typedef itk::Vector<float, ModelOrder> VectorType;     // Parameter type always float/double

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  rtk::SetProjectionsReaderFromGgo<ReaderType, args_info_rtklagcorrection>(reader, args_info);
  reader->ComputeLineIntegralOff();
  
  // Generate namefiles projections
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(args_info.nsort_flag);
  names->SetRegularExpression(args_info.regexp_arg);
  names->SetSubMatch(args_info.submatch_arg);
  reader->SetFileNames( names->GetFileNames() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update())

  VectorType a;
  a[0] = 0.7055f;
  a[1] = 0.1141f;
  a[2] = 0.0212f;
  a[3] = 0.0033f;
  
  VectorType b;
  b[0] = 2.911e-3f;
  b[1] = 0.4454e-3f;
  b[2] = 0.0748e-3f;
  b[3] = 0.0042e-3f;

#ifdef RTK_USE_CUDA
  typedef rtk::CudaLagCorrectionImageFilter LagType;
#else
  typedef rtk::LagCorrectionImageFilter<OutputImageType, ModelOrder> LagType;
#endif
  LagType::Pointer lagfilter = LagType::New();
  lagfilter->SetInput(reader->GetOutput());
  lagfilter->SetCoefficients(a, b);
  lagfilter->InPlaceOff();
  lagfilter->UpdateOutputInformation();
  
  // Streaming filter
  typedef itk::StreamingImageFilter<OutputImageType, OutputImageType> StreamerType;
  StreamerType::Pointer streamer = StreamerType::New();
  streamer->SetInput(lagfilter->GetOutput());
  streamer->SetNumberOfStreamDivisions(100);
  
  // Save corrected projections
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(args_info.output_arg);
  writer->SetInput(streamer->GetOutput()); 
  writer->UpdateOutputInformation();

  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  return EXIT_SUCCESS;
}

