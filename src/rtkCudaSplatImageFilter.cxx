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

#include "rtkCudaSplatImageFilter.h"
#include "rtkCudaSplatImageFilter.hcu"

#include <itkMacro.h>

rtk::CudaSplatImageFilter
::CudaSplatImageFilter()
{
}

void
rtk::CudaSplatImageFilter
::GPUGenerateData()
{
    int4 outputSize;
    outputSize.x = this->GetOutput()->GetLargestPossibleRegion().GetSize()[0];
    outputSize.y = this->GetOutput()->GetLargestPossibleRegion().GetSize()[1];
    outputSize.z = this->GetOutput()->GetLargestPossibleRegion().GetSize()[2];
    outputSize.w = this->GetOutput()->GetLargestPossibleRegion().GetSize()[3];

    float *pvolseries = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );
    float *pvol = *(float**)( this->GetInputVolume()->GetCudaDataManager()->GetGPUBufferPointer() );

    CUDA_splat(outputSize,
               pvol,
               pvolseries,
               m_ProjectionNumber,
               m_Weights.data_array());
}
