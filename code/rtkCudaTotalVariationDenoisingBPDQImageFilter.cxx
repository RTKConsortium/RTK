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

#include "rtkCudaTotalVariationDenoisingBPDQImageFilter.h"
#include "rtkCudaTotalVariationDenoisingBPDQImageFilter.hcu"

#include <itkMacro.h>

rtk::CudaTotalVariationDenoisingBPDQImageFilter
::CudaTotalVariationDenoisingBPDQImageFilter()
{
}

void
rtk::CudaTotalVariationDenoisingBPDQImageFilter
::GPUGenerateData()
{
    int inputSize[3];
    int outputSize[3];
    float inputSpacing[3];
    float outputSpacing[3];

    for (int i=0; i<3; i++)
      {
      inputSize[i] = this->GetInput()->GetBufferedRegion().GetSize()[i];
      outputSize[i] = this->GetOutput()->GetBufferedRegion().GetSize()[i];
      inputSpacing[i] = this->GetInput()->GetSpacing()[i];
      outputSpacing[i] = this->GetOutput()->GetSpacing()[i];

      if ((inputSize[i] != outputSize[i]) || (inputSpacing[i] != outputSpacing[i]))
        {
        std::cerr << "The CUDA Total Variation Denoising BPDQ filter can only handle input and output regions of equal size and spacing" << std::endl;
        exit(1);
        }
      }

    float *pin = *(float**)( this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer() );
    float *pout = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );

    CUDA_total_variation(inputSize,
                         inputSpacing,
                         pin,
                         pout,
                         static_cast<float>(m_Gamma),
                         static_cast<float>(m_Beta),
                         m_NumberOfIterations);
}
