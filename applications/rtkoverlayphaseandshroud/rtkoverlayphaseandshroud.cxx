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

#include "rtkoverlayphaseandshroud_ggo.h"
#include "rtkMacro.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkCSVArray2DFileReader.h>
#include <itkRGBPixel.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkNumericTraits.h>

#include <fstream>

int
main(int argc, char * argv[])
{
  GGO(rtkoverlayphaseandshroud, args_info);

  using InputPixelType = double;
  using RGBPixelType = itk::RGBPixel<unsigned char>;
  constexpr unsigned int Dimension = 2;

  using InputImageType = itk::Image<InputPixelType, Dimension>;
  using OutputImageType = itk::Image<RGBPixelType, Dimension>;

  // Read
  InputImageType::Pointer readImage;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readImage = itk::ReadImage<InputImageType>(args_info.input_arg))

  // Read signal file
  using ReaderType = itk::CSVArray2DFileReader<double>;
  auto signalReader = ReaderType::New();
  signalReader->SetFileName(args_info.signal_arg);
  signalReader->SetFieldDelimiterCharacter(';');
  signalReader->HasRowHeadersOff();
  signalReader->HasColumnHeadersOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(signalReader->Update())
  std::vector<double> signal = signalReader->GetArray2DDataObject()->GetColumn(0);

  // Locate the minima in the signal
  std::vector<bool> minima;
  minima.push_back(false);
  for (unsigned int i = 1; i < signal.size(); i++)
  {
    minima.push_back((signal[i] < signal[i - 1]));
  }

  // Create output RGB image
  auto RGBout = OutputImageType::New();
  RGBout->SetRegions(readImage->GetLargestPossibleRegion());
  RGBout->Allocate();

  // Compute min and max of shroud to scale output
  itk::ImageRegionConstIterator<InputImageType>      itIn(readImage, readImage->GetLargestPossibleRegion());
  itk::ImageRegionIteratorWithIndex<OutputImageType> itOut(RGBout, RGBout->GetLargestPossibleRegion());

  double min = itk::NumericTraits<double>::max();
  double max = -itk::NumericTraits<double>::max();
  while (!itIn.IsAtEnd())
  {
    double currentPixel = itIn.Get();
    if (currentPixel < min)
      min = currentPixel;
    if (currentPixel > max)
      max = currentPixel;
    ++itIn;
  }

  // Fill the output
  itIn.GoToBegin();
  itOut.GoToBegin();
  while (!itOut.IsAtEnd())
  {
    RGBPixelType pix;

    if (minima[itOut.GetIndex()[1]]) // If it is a minimum, draw a red pixel
    {
      pix.Fill(0);
      pix.SetRed(255);
    }
    else // Otherwise, copy the input image, scaled to [0 - 255], in all channels
    {
      pix.Fill(floor((itIn.Get() - min) * 255.0 / (max - min)));
    }

    itOut.Set(pix);

    ++itIn;
    ++itOut;
  }

  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(RGBout, args_info.output_arg))

  return EXIT_SUCCESS;
}
