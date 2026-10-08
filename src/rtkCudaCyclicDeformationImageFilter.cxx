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

#include "rtkCudaCyclicDeformationImageFilter.h"
#include "rtkCudaConstantVolumeSeriesSource.h"
#include "rtkCudaCyclicDeformationImageFilter.hcu"

#include <itkMacro.h>

namespace rtk
{

CudaCyclicDeformationImageFilter ::CudaCyclicDeformationImageFilter() = default;

void
CudaCyclicDeformationImageFilter ::GPUGenerateData()
{
  // Run the superclass method that updates all member variables
  this->Superclass::BeforeThreadedGenerateData();

  // Prepare the data to perform the linear interpolation on GPU
  itk::SizeValueType inputSize[4];
  inputSize[0] = this->GetInput()->GetBufferedRegion().GetSize()[0];
  inputSize[1] = this->GetInput()->GetBufferedRegion().GetSize()[1];
  inputSize[2] = this->GetInput()->GetBufferedRegion().GetSize()[2];
  inputSize[3] = this->GetInput()->GetBufferedRegion().GetSize()[3];
  if ((this->GetOutput()->GetRequestedRegion().GetSize()[0] != inputSize[0]) ||
      (this->GetOutput()->GetRequestedRegion().GetSize()[1] != inputSize[1]) ||
      (this->GetOutput()->GetRequestedRegion().GetSize()[2] != inputSize[2]))
  {
    itkExceptionMacro("In rtk::CudaCyclicDeformationImageFilter: the output's requested region must have the same "
                      "size as the input's buffered region on the first 3 dimensions");
  }

  float * pin = static_cast<float *>(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pout = static_cast<float *>(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());

  CUDA_linear_interpolate_along_fourth_dimension(
    inputSize, pin, pout, this->m_FrameInf, this->m_FrameSup, this->m_WeightInf, this->m_WeightSup);
}

} // namespace rtk

template class itk::CudaImageToImageFilter<
  itk::CudaImage<itk::CovariantVector<float, 3>, 4>,
  itk::CudaImage<itk::CovariantVector<float, 3>, 3>,
  rtk::CyclicDeformationImageFilter<itk::CudaImage<itk::CovariantVector<float, 3>, 4>,
                                    itk::CudaImage<itk::CovariantVector<float, 3>, 3>>>;
