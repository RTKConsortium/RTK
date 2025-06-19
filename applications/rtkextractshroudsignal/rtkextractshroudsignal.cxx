/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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

namespace rtk
{
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
} // namespace rtk

int
main(int argc, char * argv[])
{
  GGO(rtkextractshroudsignal, args_info);

  using InputPixelType = double;
  using OutputPixelType = double;
  constexpr unsigned int Dimension = 2;

  using InputImageType = itk::Image<InputPixelType, Dimension>;
  using OutputImageType = itk::Image<OutputPixelType, Dimension - 1>;

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
    auto shroudFilter = shroudFilterType::New();
    shroudFilter->SetInput(itk::ReadImage<InputImageType>(args_info.input_arg));
    shroudFilter->SetAmplitude(args_info.amplitude_arg);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(shroudFilter->Update())
    shroudSignal = shroudFilter->GetOutput();
  }
  else if (std::string(args_info.method_arg) == "Reg1D")
  {
    using shroudFilterType = rtk::Reg1DExtractShroudSignalImageFilter<InputPixelType, OutputPixelType>;
    auto shroudFilter = shroudFilterType::New();
    shroudFilter->SetInput(itk::ReadImage<InputImageType>(args_info.input_arg));
    TRY_AND_EXIT_ON_ITK_EXCEPTION(shroudFilter->Update())
    shroudSignal = shroudFilter->GetOutput();
  }
  else
  {
    std::cerr << "The specified method does not exist." << std::endl;
    return 1;
  }

  // Write output signal
  rtk::WriteSignalToTextFile(shroudSignal.GetPointer(), args_info.output_arg);

  // Process phase signal if required
  if (args_info.phase_given)
  {
    using PhaseFilter = rtk::ExtractPhaseImageFilter<OutputImageType>;
    auto phase = PhaseFilter::New();
    phase->SetInput(shroudSignal);
    phase->SetMovingAverageSize(args_info.movavg_arg);
    phase->SetUnsharpMaskSize(args_info.unsharp_arg);
    phase->SetModel((PhaseFilter::ModelType)args_info.model_arg);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(phase->Update())

    rtk::WriteSignalToTextFile(phase->GetOutput(), args_info.phase_arg);
  }

  return EXIT_SUCCESS;
}
