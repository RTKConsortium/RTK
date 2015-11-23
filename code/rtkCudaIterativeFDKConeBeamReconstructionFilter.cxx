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

#include "rtkCudaIterativeFDKConeBeamReconstructionFilter.h"

rtk::CudaIterativeFDKConeBeamReconstructionFilter
::CudaIterativeFDKConeBeamReconstructionFilter()
{
  // Create each filter which are specific for cuda
  m_DisplacedDetectorFilter = DisplacedDetectorFilterType::New();
  m_ParkerFilter = ParkerFilterType::New();
  m_FDKFilter = FDKFilterType::New();
  m_ConstantProjectionStackSource = ConstantImageSourceType::New();

  // Filter parameters
  m_DisplacedDetectorFilter->SetPadOnTruncatedSide(false);
}

void
rtk::CudaIterativeFDKConeBeamReconstructionFilter
::GPUGenerateData()
{
  CPUSuperclass::GenerateData();
}
