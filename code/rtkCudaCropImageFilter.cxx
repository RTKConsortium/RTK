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

#include "rtkCudaCropImageFilter.h"
#include "rtkCudaUtilities.hcu"
#include "rtkCudaCropImageFilter.hcu"

#include <itkMacro.h>

namespace rtk
{

CudaCropImageFilter
::CudaCropImageFilter()
{
}

// CUDA cropping call
void
CudaCropImageFilter
::GPUGenerateData()
{
  OutputImageRegionType croppedRegion = this->GetExtractionRegion();
  uint3 sz, input_sz;
  long3 idx;

  idx.x = croppedRegion.GetIndex()[0];
  idx.y = croppedRegion.GetIndex()[1];
  idx.z = croppedRegion.GetIndex()[2];
  sz.x = croppedRegion.GetSize()[0];
  sz.y = croppedRegion.GetSize()[1];
  sz.z = croppedRegion.GetSize()[2];
  input_sz.x = this->GetInput()->GetLargestPossibleRegion().GetSize()[0];
  input_sz.y = this->GetInput()->GetLargestPossibleRegion().GetSize()[1];
  input_sz.z = this->GetInput()->GetLargestPossibleRegion().GetSize()[2];

  float *pin  = *(float**)( this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pout = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );

  CUDA_crop(idx,
            sz,
            input_sz,
            pin,
            pout);
}

} // end namespace rtk
