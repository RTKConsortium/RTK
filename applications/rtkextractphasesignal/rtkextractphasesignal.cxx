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

#include "rtkextractphasesignal_ggo.h"
#include "rtkMacro.h"

#include "rtkExtractPhaseImageFilter.h"

#include <itkImageFileReader.h>
#include <itkRawImageIO.h>
#include <fstream>

template<class TSignalType>
void
WriteSignalToTextFile(TSignalType *sig, const std::string &fileName)
{
  std::ofstream ofs( fileName.c_str() );
  itk::ImageRegionConstIterator<TSignalType> it( sig, sig->GetLargestPossibleRegion() );
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    ofs << it.Get() << std::endl;
  }
  ofs.close();
}

int main(int argc, char * argv[])
{
  GGO(rtkextractphasesignal, args_info);

  typedef double InputPixelType;
  typedef double OutputPixelType;
  const unsigned int Dimension = 1;

  typedef itk::Image< InputPixelType, Dimension > InputImageType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Read
  itk::ImageFileReader<InputImageType>::Pointer reader = itk::ImageFileReader<InputImageType>::New();
  reader->SetFileName(args_info.input_arg);

  OutputImageType::Pointer signal;
  signal = reader->GetOutput();

  // Process phase signal if required
  typedef rtk::ExtractPhaseImageFilter<OutputImageType> PhaseFilter;
  PhaseFilter::Pointer phase = PhaseFilter::New();
  phase->SetInput(signal);
  phase->SetMovingAverageSize(args_info.movavg_arg);
  phase->SetUnsharpMaskSize(args_info.unsharp_arg);
  phase->SetModel((PhaseFilter::ModelType)args_info.model_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( phase->Update() )

  // Write output phase
  WriteSignalToTextFile(phase->GetOutput(), args_info.output_arg);

  return EXIT_SUCCESS;
}
