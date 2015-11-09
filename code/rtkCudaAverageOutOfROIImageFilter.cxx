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

#include "rtkCudaAverageOutOfROIImageFilter.h"
#include "rtkCudaAverageOutOfROIImageFilter.hcu"

#include <itkMacro.h>

rtk::CudaAverageOutOfROIImageFilter
::CudaAverageOutOfROIImageFilter()
{
}

void
rtk::CudaAverageOutOfROIImageFilter
::GPUGenerateData()
{
  int size[4];

  for (int i=0; i<4; i++)
    {
    size[i] = this->GetOutput()->GetBufferedRegion().GetSize()[i];
    }

    float *pin = *(float**)( this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer() );
    float *pout = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );
    float *proi = *(float**)( this->GetROI()->GetCudaDataManager()->GetGPUBufferPointer() );

    CUDA_average_out_of_ROI(size,
                             pin,
                             pout,
                             proi);

    //Transfer the ROI volume back to the CPU memory to save space on the GPU
    this->GetROI()->GetCudaDataManager()->GetCPUBufferPointer();
}
