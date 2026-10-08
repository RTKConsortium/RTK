/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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

namespace rtk
{

CudaSplatImageFilter ::CudaSplatImageFilter() = default;

void
CudaSplatImageFilter ::GPUGenerateData()
{
  itk::SizeValueType outputSize[4];
  outputSize[0] = this->GetOutput()->GetLargestPossibleRegion().GetSize()[0];
  outputSize[1] = this->GetOutput()->GetLargestPossibleRegion().GetSize()[1];
  outputSize[2] = this->GetOutput()->GetLargestPossibleRegion().GetSize()[2];
  outputSize[3] = this->GetOutput()->GetLargestPossibleRegion().GetSize()[3];

  float * pvolseries = static_cast<float *>(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pvol = static_cast<float *>(this->GetInputVolume()->GetCudaDataManager()->GetGPUBufferPointer());

  CUDA_splat(outputSize, pvol, pvolseries, m_ProjectionNumber, m_Weights.data_array());
}

} // namespace rtk

template class itk::CudaInPlaceImageFilter<
  itk::CudaImage<float, 4>,
  itk::CudaImage<float, 4>,
  rtk::SplatWithKnownWeightsImageFilter<itk::CudaImage<float, 4>, itk::CudaImage<float, 3>>>;
