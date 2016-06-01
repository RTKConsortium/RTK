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

#include "rtkCudaCyclicDeformationImageFilter.h"
#include "rtkCudaCyclicDeformationImageFilter.hcu"
#include "rtkCudaConstantVolumeSeriesSource.h"

#include <itkMacro.h>

rtk::CudaCyclicDeformationImageFilter
::CudaCyclicDeformationImageFilter()
{
}

void
rtk::CudaCyclicDeformationImageFilter
::GPUGenerateData()
{
  // Run the superclass method that updates all member variables
  this->Superclass::BeforeThreadedGenerateData();

  // Prepare the data to perform the linear interpolation on GPU
  unsigned int inputSize[4];

  for (unsigned int i=0; i<4; i++)
    inputSize[i] = this->GetInput()->GetBufferedRegion().GetSize()[i];
  for (unsigned int i=0; i<3; i++)
    {
    if(this->GetOutput()->GetRequestedRegion().GetSize()[i] != inputSize[i])
      itkExceptionMacro("In rtk::CudaCyclicDeformationImageFilter: the output's requested region must have the same size as the input's buffered region on the first 3 dimensions")
    }

  float *pin = *(float**)( this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pout = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );

  CUDA_linear_interpolate_along_fourth_dimension(inputSize, pin, pout, this->m_FrameInf, this->m_FrameSup, this->m_WeightInf, this->m_WeightSup);
}
