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

#ifndef __rtkCudaSARTConeBeamReconstructionFilter_cxx
#define __rtkCudaSARTConeBeamReconstructionFilter_cxx

#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkCudaForwardProjectionImageFilter.h"
#include "rtkCudaBackProjectionImageFilter.h"
#include "rtkCudaSARTConeBeamReconstructionFilter.h"
#include "itkImageFileWriter.h"


namespace rtk
{

  CudaSARTConeBeamReconstructionFilter
  ::CudaSARTConeBeamReconstructionFilter():
      m_ExplicitGPUMemoryManagementFlag(false)
  {
      // Create each filter which are specific for cuda
      m_ForwardProjectionFilter = ForwardProjectionFilterType::New();
      BackProjectionFilterType::Pointer p = BackProjectionFilterType::New();
      this->SetBackProjectionFilter(p.GetPointer());
      //Permanent internal connections
      m_ForwardProjectionFilter->SetInput( 0, m_ZeroMultiplyFilter->GetOutput() );
      m_SubtractFilter->SetInput(1, m_ForwardProjectionFilter->GetOutput() );
  }

  void
  CudaSARTConeBeamReconstructionFilter
  ::GenerateData()
  {
    // Init GPU memory
    if(!m_ExplicitGPUMemoryManagementFlag);
      this->InitDevice();

    // Run reconstruction
    this->Superclass::GenerateData();
    // Transfer result to CPU image
    if(!m_ExplicitGPUMemoryManagementFlag);
      this->CleanUpDevice();
  }

  void
  CudaSARTConeBeamReconstructionFilter
  ::InitDevice()
  {
    //Nothing to do, memory control internally managed
  }

  void
  CudaSARTConeBeamReconstructionFilter
  ::CleanUpDevice()
  {
    //Nothing to do, memory control internally managed
  }

} // end namespace rtk

#endif // __rtkCudaSARTConeBeamReconstructionFilter_cxx
