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
    // We use FFTW for the kernel so we need to do the same thing as in the parent
    //#if defined(USE_FFTWF)
    //  this->SetGreatestPrimeFactor(13);
    //#endif
}

void
rtk::CudaSplatImageFilter
::GenerateData()
{
    this->AllocateOutputs();

    // Get input requested region
    OutputImageType::Pointer output = this->GetOutput();
    InputImageType::Pointer input = this->GetInputVolume();

    int4 outputSize;
    outputSize.x = output->GetLargestPossibleRegion().GetSize()[0];
    outputSize.y = output->GetLargestPossibleRegion().GetSize()[1];
    outputSize.z = output->GetLargestPossibleRegion().GetSize()[2];
    outputSize.w = output->GetLargestPossibleRegion().GetSize()[3];

    CUDA_splat(outputSize,
                       input->GetBufferPointer(),
                       output->GetBufferPointer(),
                       m_ProjectionNumber,
                       m_Weights.data_array());
}
