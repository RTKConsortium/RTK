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

#include "rtktestdownsample_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkDownsampleImageFilter.h"
#include "rtkConstantImageSource.h"

#ifdef RTK_USE_CUDA
  #include "itkCudaImage.h"
#endif

int main(int argc, char * argv[])
{
  GGO(rtktestdownsample, args_info);

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
  ConstantImageSourceType::SpacingType spacing;

  unsigned int *downsamplingFactors = new unsigned int[Dimension];
  for (unsigned int d=0; d<Dimension; d++) downsamplingFactors[d]=2;

  typedef rtk::DownsampleImageFilter<OutputImageType> DownsampleFilterType;


  for (unsigned int i=0; i<args_info.niterations_arg; i++)
    {
      std::cout << "In iteration " << i << std::endl;

    // Create new random volume
    size[0] = rand() % 1024;
    size[1] = rand() % 1024;
    size[2] = rand() % 1024;

    spacing[0] = 1.;
    spacing[1] = 1.;
    spacing[2] = 1.;

    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    constantImageSource->SetOrigin( origin );
    constantImageSource->SetSpacing( spacing );
    constantImageSource->SetSize( size );
    constantImageSource->SetConstant( 0. );
    constantImageSource->Update();

    // Downsample it
    DownsampleFilterType::Pointer downsample = DownsampleFilterType::New();
    downsample->SetFactors(downsamplingFactors);
    downsample->SetInput(constantImageSource->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION( downsample->Update() );
    }


  return EXIT_SUCCESS;
}
