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

#include <itkImageFileReader.h>
#include <itkRawImageIO.h>
#include <fstream>

int main(int argc, char * argv[])
{
  GGO(rtkextractshroudsignal, args_info);

  typedef double InputPixelType;
  typedef double OutputPixelType;
  const unsigned int Dimension = 2;

  typedef itk::Image< InputPixelType, Dimension > InputImageType;
  typedef itk::Image< OutputPixelType, Dimension - 1 > OutputImageType;

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
    typedef rtk::DPExtractShroudSignalImageFilter<InputPixelType, OutputPixelType> shroudFilterType;
    shroudFilterType::Pointer shroudFilter = shroudFilterType::New();
    shroudFilter->SetInput( reader->GetOutput() );
    shroudFilter->SetAmplitude( args_info.amplitude_arg );
    shroudFilter->Update();
    shroudSignal = shroudFilter->GetOutput();
  }
  else if (std::string(args_info.method_arg) == "Reg1D")
  {
    typedef rtk::Reg1DExtractShroudSignalImageFilter<InputPixelType, OutputPixelType> shroudFilterType;
    shroudFilterType::Pointer shroudFilter = shroudFilterType::New();
    shroudFilter->SetInput( reader->GetOutput() );
    shroudFilter->Update();
    shroudSignal = shroudFilter->GetOutput();
  }
  else
  {
    std::cerr << "The specified method does not exist." << std::endl;
    return 1;
  }

  // Write
  std::ofstream ofs(args_info.output_arg);
  itk::ImageRegionConstIterator<OutputImageType> it( shroudSignal, shroudSignal->GetLargestPossibleRegion() );
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    ofs << it.Get() << std::endl;
  }
  ofs.close();

  return EXIT_SUCCESS;
}
