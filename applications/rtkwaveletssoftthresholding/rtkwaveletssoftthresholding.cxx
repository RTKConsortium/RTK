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

#include "rtkwaveletssoftthresholding_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"
#include "giftDeconstructSoftThresholdReconstructImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#if RTK_USE_CUDA
#include <itkCudaImage.h>
#endif

int main(int argc, char * argv[])
{
  GGO(rtkwaveletssoftthresholding, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

#if RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension >     OutputImageType;
#endif
  
  // Read input
  typedef itk::ImageFileReader<OutputImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( args_info.input_arg );

  // Apply wavelets soft thresholding
  typedef gift::DeconstructSoftThresholdReconstructImageFilter
      <OutputImageType> WaveletsSoftThresholdFilterType;
  WaveletsSoftThresholdFilterType::Pointer waveletsSoftThreshold = WaveletsSoftThresholdFilterType::New();
  waveletsSoftThreshold->SetInput(reader->GetOutput());
  waveletsSoftThreshold->SetThreshold(args_info.threshold_arg);
  waveletsSoftThreshold->SetNumberOfLevels(args_info.levels_arg);
  waveletsSoftThreshold->SetWaveletsOrder(args_info.order_arg);

  // Write
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput(waveletsSoftThreshold->GetOutput());

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
