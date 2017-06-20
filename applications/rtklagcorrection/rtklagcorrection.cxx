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
  reader->ComputeLineIntegralOff();   // Don't want to preprocess data
  reader->SetFileNames( rtk::GetProjectionsFileNamesFromGgo(args_info) );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->Update())

  if ((args_info.coefficients_given != ModelOrder) && (args_info.rates_given != ModelOrder))
    {
    std::cerr << "Expecting 4 lags rates and coefficients values" << std::endl;
    return EXIT_FAILURE;
    }

  VectorType a, b;
  for (unsigned int i = 0; i < ModelOrder; ++i)
    {
    a[i] = args_info.rates_arg[i];
    b[i] = args_info.coefficients_arg[i];
    }

#ifdef RTK_USE_CUDA
  typedef rtk::CudaLagCorrectionImageFilter LagType;
#else
  typedef rtk::LagCorrectionImageFilter<OutputImageType, ModelOrder> LagType;
#endif
  LagType::Pointer lagfilter = LagType::New();
  lagfilter->SetInput(reader->GetOutput());
  lagfilter->SetCoefficients(a, b);
  lagfilter->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( lagfilter->Update() )

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
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->UpdateOutputInformation() )

  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update())

  return EXIT_SUCCESS;
}

