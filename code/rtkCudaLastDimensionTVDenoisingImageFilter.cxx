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

#include "rtkCudaLastDimensionTVDenoisingImageFilter.h"
#include "rtkCudaLastDimensionTVDenoisingImageFilter.hcu"

#include <itkMacro.h>

rtk::CudaLastDimensionTVDenoisingImageFilter
::CudaLastDimensionTVDenoisingImageFilter()
{
  this->SetInPlace(true);
}

void
rtk::CudaLastDimensionTVDenoisingImageFilter
::GPUGenerateData()
{
  // Allocate the output image
  this->AllocateOutputs();

  int inputSize[4];
  int outputSize[4];

//   std::cout << "PRINTING INPUT" << std::endl;
//   this->GetInput()->Print(std::cout);
//   std::cout << "PRINTING OUTPUT" << std::endl;
//   this->GetOutput()->Print(std::cout);
  
  for (int i=0; i<4; i++)
    {
    inputSize[i] = this->GetInput()->GetBufferedRegion().GetSize()[i];
    outputSize[i] = this->GetOutput()->GetBufferedRegion().GetSize()[i];
  
    if (inputSize[i] != outputSize[i])
      {

      itkGenericExceptionMacro(<< "The CUDA Last Dimension Total Variation Denoising filter can only handle input and output regions of equal size");
      }
    }

  float *pin = *(float**)( this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pout = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );

  CUDA_total_variation_last_dimension(inputSize,
                                     pin,
                                     pout,
                                     static_cast<float>(m_Gamma),
                                     static_cast<float>(m_Beta),
                                     m_NumberOfIterations);
}
