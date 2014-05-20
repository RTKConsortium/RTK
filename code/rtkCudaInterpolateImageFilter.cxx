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
    // We use FFTW for the kernel so we need to do the same thing as in the parent
    //#if defined(USE_FFTWF)
    //  this->SetGreatestPrimeFactor(13);
    //#endif
}

void
rtk::CudaInterpolateImageFilter
::GenerateData()
{
    this->AllocateOutputs();

    // Get input requested region
    InputImageType::Pointer input = this->GetInputVolumeSeries();
    OutputImageType::Pointer output = this->GetOutput();

    int4 inputSize;
    inputSize.x = input->GetLargestPossibleRegion().GetSize()[0];
    inputSize.y = input->GetLargestPossibleRegion().GetSize()[1];
    inputSize.z = input->GetLargestPossibleRegion().GetSize()[2];
    inputSize.w = input->GetLargestPossibleRegion().GetSize()[3];

//    input->Print(std::cout);
//    output->Print(std::cout);
//    std::cout << "m_ProjectionNumber = " << m_ProjectionNumber << std::endl;
//    std::cout << m_Weights << std::endl;

    CUDA_interpolation(inputSize,
                       input->GetBufferPointer(),
                       output->GetBufferPointer(),
                       m_ProjectionNumber,
                       m_Weights.data_array());
}
