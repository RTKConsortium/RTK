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

#include "rtkCudaInterpolateImageFilter.h"
#include "rtkCudaInterpolateImageFilter.hcu"

#include <itkMacro.h>

rtk::CudaInterpolateImageFilter
::CudaInterpolateImageFilter()
{
}

void
rtk::CudaInterpolateImageFilter
::GPUGenerateData()
{
    int4 inputSize;
    inputSize.x = this->GetInputVolumeSeries()->GetBufferedRegion().GetSize()[0];
    inputSize.y = this->GetInputVolumeSeries()->GetBufferedRegion().GetSize()[1];
    inputSize.z = this->GetInputVolumeSeries()->GetBufferedRegion().GetSize()[2];
    inputSize.w = this->GetInputVolumeSeries()->GetBufferedRegion().GetSize()[3];

    float *pvolseries = *(float**)( this->GetInputVolumeSeries()->GetCudaDataManager()->GetGPUBufferPointer() );
    float *pvol = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );

    CUDA_interpolation(inputSize,
                       pvolseries,
                       pvol,
                       m_ProjectionNumber,
                       m_Weights.data_array());
}
