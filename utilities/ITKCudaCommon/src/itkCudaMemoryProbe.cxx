/*=========================================================================
 *
 *  Copyright NumFOCUS
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
#include "itkCudaMemoryProbe.h"
#include "itkCudaUtil.h"
#include <cuda_runtime_api.h>

namespace itk
{
CudaMemoryProbe ::CudaMemoryProbe()
  : ResourceProbe<CudaMemoryProbe::CudaMemoryLoadType, double>("Cuda memory", "kB")
{}

CudaMemoryProbe ::~CudaMemoryProbe() = default;

CudaMemoryProbe::CudaMemoryLoadType
CudaMemoryProbe ::GetInstantValue() const
{
  size_t free = 0;
  size_t total = 0;
  CUDA_CHECK(cudaMemGetInfo(&free, &total));
  return static_cast<CudaMemoryLoadType>((OffsetValueType(total) - OffsetValueType(free)) / 1024.);
}
} // end namespace itk
