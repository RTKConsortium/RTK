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

#include "rtkCudaConstantVolumeSource.h"
#include "rtkCudaConstantVolumeSource.hcu"

#include <itkMacro.h>

rtk::CudaConstantVolumeSource
::CudaConstantVolumeSource()
{
}

void
rtk::CudaConstantVolumeSource
::GPUGenerateData()
{
    int outputSize[3];

    for (int i=0; i<3; i++)
      {
      outputSize[i] = this->GetOutput()->GetRequestedRegion().GetSize()[i];
      }

    float *pout = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );

    CUDA_generate_constant_volume(outputSize,
                                 pout,
                                 m_Constant);
}
