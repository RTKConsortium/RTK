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

#include "rtktestupsample_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkUpsampleImageFilter.h"
#include "rtkConstantImageSource.h"

#include "itkImageFileWriter.h"

#ifdef RTK_USE_CUDA
  #include "itkCudaImage.h"
#endif

int main(int argc, char * argv[])
{
  GGO(rtktestupsample, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
#endif

  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;

  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SizeType upsampledSize;
  ConstantImageSourceType::SpacingType spacing;
  OutputImageType::IndexType index;

  unsigned int *UpsamplingFactors = new unsigned int[Dimension];
  for (unsigned int d=0; d<Dimension; d++) UpsamplingFactors[d]=2;

  typedef rtk::UpsampleImageFilter<OutputImageType> UpsampleFilterType;

  typedef itk::ImageFileWriter<OutputImageType> WriteFilterType;

  // Initialize random seed
  srand (time(NULL));

  for (unsigned int i=0; i<args_info.niterations_arg; i++)
    {
    std::cout << "In iteration " << i << std::endl;

    // Create new random volume
    size[0] = rand() % 512;
    size[1] = rand() % 512;
    size[2] = rand() % 512;

    upsampledSize[0] = size[0] * 2;
    upsampledSize[1] = size[1] * 2;
    upsampledSize[2] = size[2] * 2;

    index.Fill(0);

    spacing[0] = 1.;
    spacing[1] = 1.;
    spacing[2] = 1.;

    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    constantImageSource->SetOrigin( origin );
    constantImageSource->SetSpacing( spacing );
    constantImageSource->SetSize( size );
    constantImageSource->SetConstant( 1. );
    constantImageSource->Update();

    // Upsample it
    UpsampleFilterType::Pointer Upsample = UpsampleFilterType::New();
    Upsample->SetFactors(UpsamplingFactors);
    Upsample->SetInput(constantImageSource->GetOutput());
    Upsample->SetOutputSize(upsampledSize);
    Upsample->SetOutputIndex(index);
    TRY_AND_EXIT_ON_ITK_EXCEPTION( Upsample->Update() );

    // Write both the random image and the upsampled one
    WriteFilterType::Pointer writer = WriteFilterType::New();

    writer->SetInput(constantImageSource->GetOutput());
    writer->SetFileName("toBeUpsampled.mha");
    writer->Update();

    writer->SetInput(Upsample->GetOutput());
    writer->SetFileName("upsampled.mha");
    writer->Update();

    }


  return EXIT_SUCCESS;
}
