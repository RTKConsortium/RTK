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

#include "rtkextractshroudsignal_ggo.h"
#include "rtkMacro.h"

#include "rtkDPExtractShroudSignalImageFilter.h"
#include "rtkReg1DExtractShroudSignalImageFilter.h"
#include "rtkExtractPhaseImageFilter.h"

#include <itkImageFileReader.h>
#include <fstream>

template <class TSignalType>
void
WriteSignalToTextFile(TSignalType * sig, const std::string & fileName)
{
  std::ofstream                              ofs(fileName.c_str());
  itk::ImageRegionConstIterator<TSignalType> it(sig, sig->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    ofs << it.Get() << std::endl;
  }
  ofs.close();
}

int
main(int argc, char * argv[])
{
  GGO(rtkextractshroudsignal, args_info);

  using InputPixelType = double;
  using OutputPixelType = double;
  constexpr unsigned int Dimension = 2;

  using InputImageType = itk::Image<InputPixelType, Dimension>;
  using OutputImageType = itk::Image<OutputPixelType, Dimension - 1>;

  // Read
  itk::ImageFileReader<InputImageType>::Pointer reader = itk::ImageFileReader<InputImageType>::New();
  reader->SetFileName(args_info.input_arg);

  // Extract shroud signal
  OutputImageType::Pointer shroudSignal;
  if (std::string(args_info.method_arg) == "DynamicProgramming")
  {
    if (!args_info.amplitude_given)
    {
      std::cerr << "You must supply a maximum amplitude to look for." << std::endl;
      return 1;
    }
    using shroudFilterType = rtk::DPExtractShroudSignalImageFilter<InputPixelType, OutputPixelType>;
    shroudFilterType::Pointer shroudFilter = shroudFilterType::New();
    shroudFilter->SetInput(reader->GetOutput());
    shroudFilter->SetAmplitude(args_info.amplitude_arg);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(shroudFilter->Update())
    shroudSignal = shroudFilter->GetOutput();
  }
  else if (std::string(args_info.method_arg) == "Reg1D")
  {
    using shroudFilterType = rtk::Reg1DExtractShroudSignalImageFilter<InputPixelType, OutputPixelType>;
    shroudFilterType::Pointer shroudFilter = shroudFilterType::New();
    shroudFilter->SetInput(reader->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(shroudFilter->Update())
    shroudSignal = shroudFilter->GetOutput();
  }
  else
  {
    std::cerr << "The specified method does not exist." << std::endl;
    return 1;
  }

  // Write output signal
  WriteSignalToTextFile(shroudSignal.GetPointer(), args_info.output_arg);

  // Process phase signal if required
  if (args_info.phase_given)
  {
    using PhaseFilter = rtk::ExtractPhaseImageFilter<OutputImageType>;
    PhaseFilter::Pointer phase = PhaseFilter::New();
    phase->SetInput(shroudSignal);
    phase->SetMovingAverageSize(args_info.movavg_arg);
    phase->SetUnsharpMaskSize(args_info.unsharp_arg);
    phase->SetModel((PhaseFilter::ModelType)args_info.model_arg);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(phase->Update())

    WriteSignalToTextFile(phase->GetOutput(), args_info.phase_arg);
  }

  return EXIT_SUCCESS;
}
