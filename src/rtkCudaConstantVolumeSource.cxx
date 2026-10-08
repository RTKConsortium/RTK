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

#include "rtkCudaConstantVolumeSource.h"
#include "rtkCudaConstantVolumeSource.hcu"

#include <itkMacro.h>

namespace rtk
{

CudaConstantVolumeSource ::CudaConstantVolumeSource() = default;

void
CudaConstantVolumeSource ::GPUGenerateData()
{
  itk::SizeValueType outputSize[3];
  outputSize[0] = this->GetOutput()->GetRequestedRegion().GetSize()[0];
  outputSize[1] = this->GetOutput()->GetRequestedRegion().GetSize()[1];
  outputSize[2] = this->GetOutput()->GetRequestedRegion().GetSize()[2];

  float * pout = static_cast<float *>(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());

  CUDA_generate_constant_volume(outputSize, pout, m_Constant);
}

} // namespace rtk

template class itk::CudaImageToImageFilter<itk::CudaImage<float, 3>,
                                           itk::CudaImage<float, 3>,
                                           rtk::ConstantImageSource<itk::CudaImage<float, 3>>>;
