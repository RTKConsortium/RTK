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

#ifndef __rtkOpenCLFDKConeBeamReconstructionFilter_hxx
#define __rtkOpenCLFDKConeBeamReconstructionFilter_hxx

namespace rtk
{

OpenCLFDKConeBeamReconstructionFilter
::OpenCLFDKConeBeamReconstructionFilter()
{
  // Create each filter which are specific for OpenCL
  m_BackProjectionFilter = BackProjectionFilterType::New();

  //Permanent internal connections
  m_BackProjectionFilter->SetInput( 1, m_RampFilter->GetOutput() );

  // Default parameters
  m_BackProjectionFilter->InPlaceOn();
  m_BackProjectionFilter->SetTranspose(false);
}

void
OpenCLFDKConeBeamReconstructionFilter
::GenerateData()
{
  BackProjectionFilterType* openclbp = dynamic_cast<BackProjectionFilterType*>( m_BackProjectionFilter.GetPointer() );

  // Init GPU memory
  openclbp->InitDevice();

  // Run reconstruction
  this->Superclass::GenerateData();

  // Transfer result to CPU image
  openclbp->CleanUpDevice();
}

} // end namespace rtk

#endif // __rtkOpenCLFDKConeBeamReconstructionFilter_hxx
