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

#include "rtkCudaAverageOutOfROIImageFilter.h"
#include "rtkCudaAverageOutOfROIImageFilter.hcu"

#include <itkMacro.h>

namespace rtk
{

CudaAverageOutOfROIImageFilter ::CudaAverageOutOfROIImageFilter() = default;

void
CudaAverageOutOfROIImageFilter ::GPUGenerateData()
{
  itk::SizeValueType size[4];
  size[0] = this->GetOutput()->GetBufferedRegion().GetSize()[0];
  size[1] = this->GetOutput()->GetBufferedRegion().GetSize()[1];
  size[2] = this->GetOutput()->GetBufferedRegion().GetSize()[2];
  size[3] = this->GetOutput()->GetBufferedRegion().GetSize()[3];

  float * pin = static_cast<float *>(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pout = static_cast<float *>(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());
  float * proi = static_cast<float *>(this->GetROI()->GetCudaDataManager()->GetGPUBufferPointer());

  CUDA_average_out_of_ROI(size, pin, pout, proi);

  // Transfer the ROI volume back to the CPU memory to save space on the GPU
  this->GetROI()->GetCudaDataManager()->GetCPUBufferPointer();
}

} // namespace rtk

template class itk::CudaInPlaceImageFilter<
  itk::CudaImage<float, 4>,
  itk::CudaImage<float, 4>,
  rtk::AverageOutOfROIImageFilter<itk::CudaImage<float, 4>, itk::CudaImage<float, 3>>>;
