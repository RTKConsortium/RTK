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

#include "rtkcheckimagequality_ggo.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionConstIterator.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"

namespace rtk
{
/**
 * \file rtkcheckimagequality.cxx
 *
 * \brief Checks that an image has a satisfactory MSE against a reference.
 *
 * \author Aurélien Coussat
 */

template <class TImage>
double
MSE(const typename TImage::Pointer & reference, const typename TImage::Pointer & reconstruction)
{
  using ImageIteratorType = itk::ImageRegionConstIterator<TImage>;
  ImageIteratorType itTest(reconstruction, reconstruction->GetBufferedRegion());
  ImageIteratorType itRef(reference, reference->GetBufferedRegion());

  using ErrorType = double;
  ErrorType EnerError = 0.;

  itTest.GoToBegin();
  itRef.GoToBegin();

  while (!itRef.IsAtEnd())
  {
    typename TImage::PixelType TestVal = itTest.Get();
    typename TImage::PixelType RefVal = itRef.Get();
    EnerError += std::pow(ErrorType(RefVal - TestVal), 2.);
    ++itTest;
    ++itRef;
  }

  return EnerError;
}
} // namespace rtk

int
main(int argc, char ** argv)
{
  GGO(rtkcheckimagequality, args_info);

  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, Dimension>;

  // Maximum number of comparisons to perform (depends on the number of inputs)
  unsigned int n_max =
    std::max({ args_info.reference_given, args_info.reconstruction_given, args_info.threshold_given });

  for (unsigned int i = 0; i < n_max; i++)
  {
    unsigned int reference_index = std::min(args_info.reference_given - 1, i);
    unsigned int reconstruction_index = std::min(args_info.reconstruction_given - 1, i);
    unsigned int threshold_index = std::min(args_info.threshold_given - 1, i);

    ImageType::Pointer reference, reconstruction;
    try
    {
      reference = itk::ReadImage<ImageType>(args_info.reference_arg[reference_index]);
    }
    catch (::itk::ExceptionObject & e)
    {
      std::cerr << e.GetDescription();
      return EXIT_FAILURE;
    }

    try
    {
      reconstruction = itk::ReadImage<ImageType>(args_info.reconstruction_arg[reconstruction_index]);
    }
    catch (::itk::ExceptionObject & e)
    {
      std::cerr << e.GetDescription();
      return EXIT_FAILURE;
    }

    double mse = rtk::MSE<ImageType>(reference, reconstruction);

    if (mse > args_info.threshold_arg[threshold_index])
    {
      std::cerr << "Error comparing " << args_info.reference_arg[reference_index] << " and "
                << args_info.reconstruction_arg[reconstruction_index] << ":" << std::endl
                << "MSE " << mse << " above given threshold " << args_info.threshold_arg[threshold_index] << std::endl;
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
