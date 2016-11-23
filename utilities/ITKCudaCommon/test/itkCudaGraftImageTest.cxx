/*=========================================================================
*
*  Copyright Insight Software Consortium
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

/**
 * Test program for itkCudaImage class.
 * This program shows how to use Cuda image and Cuda program.
 */
#include "itkCudaImage.h"
#include "itkInPlaceImageTestFilter.h"

int main(int argc, char *argv[])
{
  
  typedef itk::CudaImage<float,2> ImageType;
  
  ImageType::Pointer cudaImage = ImageType::New();
  
  ImageType::IndexType index;
  index[0] = 10;
  index[1] = 10;
 
  ImageType::Pointer test2 = ImageType::New();
  // Allocate the image
  {
  ImageType::RegionType region;
  ImageType::SizeType size;
  size[0] = 200;
  size[1] = 200;
  region.SetSize(size);
  cudaImage->SetRegions(region);
  cudaImage->Allocate();
  cudaImage->FillBuffer(3.0);

  cudaImage->UpdateBuffers();
  std::cout << cudaImage->GetPixel(index) << std::endl;
  
  ImageType::Pointer test = cudaImage;
  test2->Graft(cudaImage);
  test2->GetCudaDataManager()->SetCPUBufferDirty();
  }

  std::cout << test2->GetPixel(index) << std::endl;
  test2->UpdateBuffers();
  
  typedef itk::InPlaceImageTestFilter<ImageType,ImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(test2);
  filter->Update();

  std::cout << "Test DONE!" << std::endl;
  return EXIT_SUCCESS;
}
