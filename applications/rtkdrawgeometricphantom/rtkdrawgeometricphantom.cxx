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

#include "rtkdrawgeometricphantom_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkDrawGeometricPhantomImageFilter.h"
#include "itkImageFileWriter.h"
#include "rtkAdditiveGaussianNoiseImageFilter.h"


int
main(int argc, char * argv[])
{
  GGO(rtkdrawgeometricphantom, args_info);

  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  // Empty volume image
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  auto constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkdrawgeometricphantom>(constantImageSource,
                                                                                                 args_info);

  using DQType = rtk::DrawGeometricPhantomImageFilter<OutputImageType, OutputImageType>;

  // Offset, scale, rotation
  DQType::VectorType offset(0.);
  if (args_info.offset_given)
  {
    if (args_info.offset_given > 3)
    {
      std::cerr << "--offset needs up to 3 values" << std::endl;
      exit(EXIT_FAILURE);
    }
    offset[0] = args_info.offset_arg[0];
    offset[1] = args_info.offset_arg[1];
    offset[2] = args_info.offset_arg[2];
  }
  DQType::VectorType scale;
  scale.Fill(args_info.phantomscale_arg[0]);
  if (args_info.phantomscale_given)
  {
    if (args_info.phantomscale_given > 3)
    {
      std::cerr << "--phantomscale needs up to 3 values" << std::endl;
      exit(EXIT_FAILURE);
    }
    for (unsigned int i = 0; i < std::min(args_info.phantomscale_given, Dimension); i++)
      scale[i] = args_info.phantomscale_arg[i];
  }
  DQType::RotationMatrixType rot;
  rot.SetIdentity();
  if (args_info.rotation_given)
  {
    if (args_info.rotation_given != 9)
    {
      std::cerr << "--rotation needs exactly 9 values" << std::endl;
      exit(EXIT_FAILURE);
    }
    for (unsigned int i = 0; i < Dimension; i++)
      for (unsigned int j = 0; j < Dimension; j++)
        rot[i][j] = args_info.rotation_arg[i * Dimension + j];
  }

  // Reference
  if (args_info.verbose_flag)
    std::cout << "Creating reference... " << std::flush;
  auto dq = DQType::New();
  dq->SetInput(constantImageSource->GetOutput());
  dq->SetPhantomScale(scale);
  dq->SetOriginOffset(offset);
  dq->SetRotationMatrix(rot);
  dq->SetConfigFile(args_info.phantomfile_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dq->Update())

  // Add noise
  OutputImageType::Pointer output = dq->GetOutput();
  if (args_info.noise_given)
  {
    using NIFType = rtk::AdditiveGaussianNoiseImageFilter<OutputImageType>;
    auto noisy = NIFType::New();
    noisy->SetInput(output);
    noisy->SetMean(0.0);
    noisy->SetStandardDeviation(args_info.noise_arg);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(noisy->Update())
    output = noisy->GetOutput();
  }

  // Write
  if (args_info.verbose_flag)
    std::cout << "Writing reference... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(output, args_info.output_arg))

  return EXIT_SUCCESS;
}
